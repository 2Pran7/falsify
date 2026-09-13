"""The four tools: schemas, validation, dispatch. THIS FILE IS THE BOUNDARY.

Everything the model is allowed to do to the pipeline passes through `dispatch`.
Everything the model is allowed to see comes back from it. That makes this the
one file where "the LLM decides, never computes" is either true or false.

Four rules the code here enforces, in the order they matter:

  1. THE MENU IS CLOSED. The model selects a tool by name and a feature by name
     from fixed sets. There is no path from model output to executed code: no
     eval, no exec, no getattr on a model-supplied string. A plan is data.

  2. ARGUMENTS ARE VALIDATED BEFORE DISPATCH. Unknown keys, missing required
     keys and wrong types are rejected without the pipeline being touched. A
     model that invents a parameter gets a usable error, not a TypeError from
     three frames down.

  3. RESULTS CROSS AS SUMMARIES, NEVER PAYLOADS. Real outputs go into the
     Session; the model receives a handle and a digest under the size cap. It
     cannot average a series it has never seen.

  4. ERRORS COME BACK AS DATA. Every failure becomes a ToolError carrying a
     message written for the model, because the loop hands it back as a
     tool_result and the next turn is the model's chance to correct itself. An
     unhandled exception kills the run and wastes everything spent so far.

The fourth rule has a cost consequence worth stating: a bad error message is
paid for in turns, and turns are the budget.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from falsify.agent.session import Session, SessionError
from falsify.backtest import metrics as m
from falsify.backtest.engine import BacktestConfig
from falsify.backtest.engine import run_backtest as engine_run_backtest
from falsify.backtest.loader import load_panel
from falsify.backtest.portfolio import (
    decile_weights,
    hold_until_next_rebalance,
    month_end_dates,
)
from falsify.features import library as feat
from falsify.stats import deflated

# The closed menus. A name outside these sets never reaches the pipeline.
FEATURES: dict[str, str] = {
    "mom_12_1": "Jegadeesh-Titman 12-1 momentum: the 12-month return skipping "
                "the most recent month. Needs 252 trading days of history per "
                "ticker before it produces any value.",
    "vol_21d": "Annualised volatility of daily log returns over 21 trading days.",
    "vol_63d": "Annualised volatility of daily log returns over 63 trading days.",
    "ret_1d": "Simple one-day return.",
    "ret_5d": "Simple five-day return.",
    "ret_21d": "Simple 21-day (one month) return.",
    "sma_50d": "50-day simple moving average of the close.",
    "sma_200d": "200-day simple moving average of the close.",
}

REBALANCE_FREQUENCIES: tuple[str, ...] = ("monthly",)

MIN_BUCKETS, MAX_BUCKETS = 2, 20
MAX_COST_BPS = 100.0

# Variance of the estimated Sharpes ACROSS trials, needed by the deflation
# benchmark. This is an ASSUMPTION until the six anomalies at M6 supply a
# measured value, and every output carries it as a labelled field so no number
# leaves this module pretending the input was observed.
TRIAL_VARIANCE = 0.0009


class ToolError(Exception):
    """A failure the model is expected to read and recover from.

    Never let a raw exception escape `dispatch`. The loop turns a ToolError
    into an error tool_result and the conversation continues; anything else
    ends the run and throws away every token already spent on it.
    """


# The feature menu, resolved to pipeline calls. A dict of pre-bound callables,
# NOT a getattr on a model-supplied name: the model picks a key that already
# exists, it never names a function to be looked up. That distinction is the
# whole of rule 1.
_BUILDERS = {
    "mom_12_1": lambda df: feat.add_momentum_12_1(df),
    "vol_21d": lambda df: feat.add_rolling_vol(df, 21),
    "vol_63d": lambda df: feat.add_rolling_vol(df, 63),
    "ret_1d": lambda df: feat.add_returns(df, 1),
    "ret_5d": lambda df: feat.add_returns(df, 5),
    "ret_21d": lambda df: feat.add_returns(df, 21),
    "sma_50d": lambda df: feat.add_sma(df, 50),
    "sma_200d": lambda df: feat.add_sma(df, 200),
}

# One table drives the schemas AND the validation, so the two cannot drift.
# A schema that advertises a parameter the validator rejects is a bug the model
# pays for in wasted turns, and keeping them in separate places is how that bug
# happens.
_SPECS: dict[str, dict[str, Any]] = {
    "fetch_data": {
        "description": (
            "Load daily price bars from the database into a stored panel. Returns a HANDLE "
            "plus a shape summary, never the rows themselves: a full universe over two years "
            "is roughly 275,000 rows. Pass the handle to compute_feature next. Call with no "
            "arguments to load every ingested ticker over the full available history."
        ),
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Uppercase tickers to restrict to. Omit for the whole universe.",
            },
            "start": {"type": "string", "description": "Inclusive ISO start date, e.g. 2024-09-04."},
            "end": {"type": "string", "description": "Inclusive ISO end date."},
        },
        "required": [],
        "types": {"tickers": list, "start": str, "end": str},
    },
    "compute_feature": {
        "description": (
            "Append one feature column to a stored price panel and return a new handle. "
            "Reports coverage, which matters: mom_12_1 needs 252 trading days of history per "
            "ticker, so on a two-year panel the first year scores nothing at all. Check "
            "n_non_null against n_rows before drawing any conclusion from a backtest built "
            "on this feature."
        ),
        "properties": {
            "panel_handle": {"type": "string", "description": "A handle returned by fetch_data."},
            "feature": {
                "type": "string",
                "enum": sorted(FEATURES),
                "description": "Which feature to compute. "
                + " ".join(f"{k}: {v}" for k, v in FEATURES.items()),
            },
        },
        "required": ["panel_handle", "feature"],
        "types": {"panel_handle": str, "feature": str},
    },
    "run_backtest": {
        "description": (
            "Sort the cross-section on a stored feature, build monthly-rebalanced portfolio "
            "weights, and run the backtest engine. Returns a handle plus performance metrics "
            "over the INVESTED window only, because a signal needing 252 days of history "
            "leaves the early sample holding nothing and averaging over those flat days "
            "understates every metric. Every call counts as a trial for later deflation. long_short=True is the dollar-neutral spread that isolates the cross-sectional effect and is the correct test of a ranking hypothesis; long_short=False holds the market plus a tilt, so its Sharpe includes market beta that no tool here can separate out."
        ),
        "properties": {
            "feature_handle": {
                "type": "string",
                "description": "A handle returned by compute_feature.",
            },
            "n_buckets": {
                "type": "integer",
                "description": f"Cross-sectional buckets, {MIN_BUCKETS} to {MAX_BUCKETS}. "
                               "10 = deciles.",
            },
            "long_short": {
                "type": "boolean",
                "description": "True for dollar-neutral long top bucket / short bottom. "
                               "False for long-only.",
            },
            "rebalance": {
                "type": "string",
                "enum": list(REBALANCE_FREQUENCIES),
                "description": "Rebalancing frequency.",
            },
            "cost_bps": {
                "type": "number",
                "description": f"One-way transaction cost in basis points, 0 to {MAX_COST_BPS}.",
            },
        },
        "required": ["feature_handle"],
        "types": {
            "feature_handle": str,
            "n_buckets": int,
            "long_short": bool,
            "rebalance": str,
            "cost_bps": (int, float),
        },
    },
    "analyze_results": {
        "description": (
            "Apply the statistical rigour layer to a stored backtest: probabilistic Sharpe, "
            "the expected best-of-N-trials Sharpe, the deflated Sharpe, and the minimum track "
            "record length. This is what decides whether a Sharpe ratio means anything. "
            "n_trials defaults to the number of backtests actually run in this session, which "
            "is the honest count; supplying a lower number will not reduce the deflation. "
            "IMPORTANT: the count is taken when you call, so an analysis run before further "
            "backtests is deflated too generously. After your LAST backtest, call this again "
            "on every backtest you intend to report, and quote only those numbers. The "
            "prob_* fields it returns are probabilities in [0,1], never Sharpe ratios."
        ),
        "properties": {
            "backtest_handle": {
                "type": "string",
                "description": "A handle returned by run_backtest.",
            },
            "n_trials": {
                "type": "integer",
                "description": "Honest number of strategies tried. Omit to use the session's "
                               "own count.",
            },
        },
        "required": ["backtest_handle"],
        "types": {"backtest_handle": str, "n_trials": int},
    },
}


def tool_schemas() -> list[dict[str, Any]]:
    """The four tool definitions, in Anthropic tool-use format.

    Returns:
        A list of dicts with keys `name`, `description` and `input_schema`,
        ready to pass as `tools=` to `messages.create`.

    These descriptions are the highest-leverage text in Module 4. They are the
    only place the model learns that `mom_12_1` needs 252 days of history, that
    a handle from `fetch_data` is what `compute_feature` consumes, or that a
    233-day sample cannot support a conclusion. A vague description is the
    single most common cause of agent failure.
    """
    return [
        {
            "name": name,
            "description": spec["description"],
            "input_schema": {
                "type": "object",
                "properties": spec["properties"],
                "required": spec["required"],
            },
        }
        for name, spec in _SPECS.items()
    ]


def validate_arguments(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Check model-supplied arguments against the tool's schema.

    Args:
        tool_name: which tool the model asked for.
        arguments: the raw `input` dict from the model's tool_use block.

    Returns:
        The arguments, unchanged, if every check passes.

    Raises:
        ToolError: unknown tool, unknown argument key, missing required
            argument, wrong type, or a value outside its allowed set or range.

    Checked in that order deliberately: report the unknown tool before
    complaining about its arguments, or the model fixes the arguments of a tool
    that does not exist.

    Unknown KEYS are rejected rather than ignored. A model that sends
    `n_deciles` when the parameter is `n_buckets` has misunderstood something,
    and silently dropping the key runs a backtest with a default the model did
    not choose and will then reason about as though it had.
    """
    spec = _SPECS.get(tool_name)
    if spec is None:
        raise ToolError(f"unknown tool {tool_name!r}. Valid tools: {', '.join(_SPECS)}")

    unknown = set(arguments) - set(spec["properties"])
    if unknown:
        raise ToolError(
            f"{tool_name}: unknown argument(s) {', '.join(sorted(unknown))}. "
            f"Accepted: {', '.join(spec['properties'])}"
        )

    missing = set(spec["required"]) - set(arguments)
    if missing:
        raise ToolError(
            f"{tool_name}: missing required argument(s) {', '.join(sorted(missing))}"
        )

    for key, value in arguments.items():
        want = spec["types"][key]
        # bool is a subclass of int in Python, so isinstance(True, int) is True.
        # Without this, `n_buckets: true` would sail through and be used as 1.
        if want is int and isinstance(value, bool):
            raise ToolError(f"{tool_name}.{key}: expected integer, got boolean")
        if not isinstance(value, want):
            raise ToolError(
                f"{tool_name}.{key}: expected {want}, got {type(value).__name__}"
            )

    # Value-level checks. Ranges are enforced here rather than trusted to the
    # JSON schema, because the schema is advice to the model and this is the
    # gate: a model can and will send something outside it.
    if tool_name == "compute_feature" and arguments["feature"] not in FEATURES:
        raise ToolError(
            f"unknown feature {arguments['feature']!r}. "
            f"Available: {', '.join(sorted(FEATURES))}"
        )
    if tool_name == "run_backtest":
        n = arguments.get("n_buckets", 10)
        if not MIN_BUCKETS <= n <= MAX_BUCKETS:
            raise ToolError(
                f"n_buckets must be between {MIN_BUCKETS} and {MAX_BUCKETS}, got {n}"
            )
        c = arguments.get("cost_bps", 10.0)
        if not 0.0 <= c <= MAX_COST_BPS:
            raise ToolError(f"cost_bps must be between 0 and {MAX_COST_BPS}, got {c}")
        r = arguments.get("rebalance", "monthly")
        if r not in REBALANCE_FREQUENCIES:
            raise ToolError(f"rebalance must be one of {REBALANCE_FREQUENCIES}, got {r!r}")
    if tool_name == "analyze_results" and "n_trials" in arguments:
        if arguments["n_trials"] < 1:
            raise ToolError("n_trials must be at least 1")

    return arguments


def dispatch(tool_name: str, arguments: dict[str, Any], session: Session) -> dict[str, Any]:
    """Validate, execute, and return the model-visible result.

    Args:
        tool_name: from the model's tool_use block.
        arguments: from the model's tool_use block.
        session: the run's handle store.

    Returns:
        A JSON-safe dict small enough to sit in the context window. Always
        includes a `handle` where the tool produced a stored artifact.

    Raises:
        ToolError: for every failure mode, including ones raised deeper in the
            pipeline.

    This function must never let a non-ToolError exception escape. The loop
    turns a ToolError into an error tool_result and the conversation carries on;
    anything else ends a run that has already been paid for.
    """
    validate_arguments(tool_name, arguments)

    impl = {
        "fetch_data": fetch_data,
        "compute_feature": compute_feature,
        "run_backtest": run_backtest,
        "analyze_results": analyze_results,
    }[tool_name]

    try:
        return impl(session, **arguments)
    except ToolError:
        raise
    except SessionError as exc:
        # Already written for the model: it names the valid handles.
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        # The safety net. The tools catch what they expect; this catches what
        # nobody anticipated, so one unforeseen exception cannot end the run.
        raise ToolError(f"{tool_name} failed: {type(exc).__name__}: {exc}") from exc


# --- the four tools --------------------------------------------------------


def fetch_data(
    session: Session,
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    """Load a price panel from the database and store it.

    Args:
        session: the run's handle store.
        tickers: restrict to these tickers; None loads every ingested ticker.
        start, end: inclusive ISO date bounds; None is unbounded.

    Returns:
        `handle` plus a digest: n_rows, n_tickers, first_date, last_date.

    Raises:
        ToolError: the query returned nothing, which the model must be told
            explicitly — an empty panel silently produces an empty backtest and
            a confident conclusion about nothing.

    The panel itself goes into the Session and never into the return value.
    548 tickers over two years is ~275,000 rows.
    """
    panel = load_panel(tickers, start, end)
    if panel.is_empty():
        raise ToolError(
            "no rows matched. Check the tickers are ingested and the date range is covered."
        )
    summary = {
        "n_rows": len(panel),
        "n_tickers": panel["ticker"].n_unique(),
        "first_date": str(panel["ts"].min()),
        "last_date": str(panel["ts"].max()),
    }
    return {"handle": session.put("panel", panel, summary), **summary}


def compute_feature(session: Session, panel_handle: str, feature: str) -> dict[str, Any]:
    """Append one feature column to a stored panel.

    Args:
        session: the run's handle store.
        panel_handle: a handle from `fetch_data`.
        feature: a key of FEATURES. Anything else is rejected.

    Returns:
        `handle` for the feature frame, plus coverage: n_rows, n_non_null,
        n_tickers_with_values, first_date_with_values.

    Raises:
        ToolError: unknown feature, bad handle, or a feature that produced no
            non-null values at all.

    Coverage is reported because it is how the model learns that `mom_12_1` on
    a two-year panel is null for the first year. Without it the model sees a
    successful call and reasons as though the whole sample were scored.
    """
    if feature not in FEATURES:
        raise ToolError(
            f"unknown feature {feature!r}. Available: {', '.join(sorted(FEATURES))}"
        )
    try:
        panel = session.payload(panel_handle, "panel")
    except SessionError as exc:
        raise ToolError(str(exc)) from exc

    frame = _BUILDERS[feature](panel)
    col = frame[feature]
    n_non_null = int(col.len() - col.null_count())
    if n_non_null == 0:
        raise ToolError(
            f"{feature} produced no values on this panel. It needs more history per "
            f"ticker than the {len(panel['ts'].unique())} trading days available."
        )

    scored = frame.drop_nulls(feature)
    summary = {
        "feature": feature,
        "n_rows": len(frame),
        "n_non_null": n_non_null,
        "n_tickers_with_values": scored["ticker"].n_unique(),
        "first_date_with_values": str(scored["ts"].min()),
    }
    return {"handle": session.put("feature", frame, summary), **summary}


def run_backtest(
    session: Session,
    feature_handle: str,
    n_buckets: int = 10,
    long_short: bool = True,
    rebalance: str = "monthly",
    cost_bps: float = 10.0,
) -> dict[str, Any]:
    """Sort on the feature, build weights, and run the engine.

    Args:
        session: the run's handle store.
        feature_handle: a handle from `compute_feature`.
        n_buckets: cross-sectional buckets, 2 to 20.
        long_short: long the top bucket and short the bottom (dollar-neutral),
            or long the top bucket only.
        rebalance: currently only "monthly".
        cost_bps: one-way transaction cost in basis points, 0 to 100.

    Returns:
        `handle` plus `metrics.summary` over the INVESTED window, plus
        n_invested_days and the trial count so far this run.

    Raises:
        ToolError: bad handle, out-of-range parameters, or a feature that
            scores no dates.

    Metrics are reported over the invested window, not the full sample. The
    signal needs 252 days of history, so the early sample holds no positions,
    and averaging over those flat days understates every metric while looking
    perfectly reasonable.
    """
    try:
        art = session.get(feature_handle, "feature")
    except SessionError as exc:
        raise ToolError(str(exc)) from exc

    frame, col = art.payload, art.summary["feature"]

    signal = frame.select(["ts", "ticker", pl.col(col).alias("sig")]).drop_nulls("sig")
    if signal.is_empty():
        raise ToolError(f"{col} scores no dates; cannot rank a cross-section.")

    rebal = pl.DataFrame({"ts": month_end_dates(signal["ts"])})
    targets = decile_weights(signal.join(rebal, on="ts", how="semi"), n_buckets, long_short)
    weights = hold_until_next_rebalance(targets, frame["ts"].unique())
    if weights.is_empty():
        raise ToolError("no rebalance date carries a scored cross-section.")

    res = engine_run_backtest(
        frame.select(["ticker", "ts", "close"]),
        weights,
        BacktestConfig(cost_bps=cost_bps),
    )

    # The invested window: dates the strategy actually held positions. Metrics
    # over the full sample would average in a year of flat pre-signal days.
    invested_dates = weights.select("ts").unique()
    invested = res.returns.join(invested_dates, on="ts", how="semi").sort("ts")

    summary = {
        **{k: round(v, 6) for k, v in m.summary(invested["ret"]).items()},
        "feature": col,
        "n_buckets": n_buckets,
        "long_short": long_short,
        "cost_bps": cost_bps,
        "n_invested_days": len(invested),
        "n_names_traded": weights["ticker"].n_unique(),
    }
    handle = session.put("backtest", {"result": res, "invested": invested}, summary)
    return {"handle": handle, **summary, "n_trials_so_far": session.n_backtests}


def analyze_results(
    session: Session, backtest_handle: str, n_trials: int | None = None
) -> dict[str, Any]:
    """Apply the Module 3 statistics to a stored backtest.

    Args:
        session: the run's handle store.
        backtest_handle: a handle from `run_backtest`.
        n_trials: the honest number of strategies tried. None uses the count
            the Session has observed, which is the safe default.

    Returns:
        Per-period and annualised Sharpe, PSR against zero, E[max SR] for the
        trial count, the deflated Sharpe, the minimum track record length, and
        the naive p-value. Plus `n_trials_used` and `n_trials_observed`.

    Raises:
        ToolError: bad handle, or a return series too short for the statistics.

    THE ANTI-CHEAT: if the model supplies an `n_trials` BELOW the number of
    backtests the Session has actually seen, the Session's count wins and the
    substitution is reported in the output. Understating N is how a deflated
    Sharpe gets quietly re-inflated, and it is the easiest place in the whole
    project to fool yourself. A model summarising its own work has every
    incentive to forget the variants it abandoned, so the count comes from the
    side of the boundary that cannot be talked out of it.

    A HIGHER n_trials than observed is accepted as given: the model may know
    about trials from outside this run, and erring toward more deflation is the
    conservative direction.
    """
    try:
        payload = session.payload(backtest_handle, "backtest")
    except SessionError as exc:
        raise ToolError(str(exc)) from exc

    returns = payload["invested"]["ret"]

    observed = session.n_backtests
    requested = n_trials
    used = observed if n_trials is None else max(n_trials, observed)
    overridden = requested is not None and used != requested

    try:
        st = deflated.sharpe_stats(returns)
        psr = deflated.probabilistic_sharpe(st["sr"], st["skew"], st["kurt"], int(st["n"]))
        emax = deflated.expected_max_sharpe(used, TRIAL_VARIANCE)
        dsr = deflated.deflated_sharpe(returns, used, TRIAL_VARIANCE)
        trl = deflated.min_track_record_length(st["sr"], st["skew"], st["kurt"])
    except ValueError as exc:
        raise ToolError(
            f"series too short or degenerate for the statistics: {exc}"
        ) from exc

    out = {
        "sharpe_per_period": round(st["sr"], 6),
        "sharpe_annualised": round(st["sr"] * (252**0.5), 4),
        "skew": round(st["skew"], 4),
        "kurtosis": round(st["kurt"], 4),
        "n_observations": int(st["n"]),
        # PSR and DSR are PROBABILITIES, not Sharpe ratios. The field names say
        # so, because a key called "deflated_sharpe" invites the reader to put
        # a number in [0,1] next to an actual Sharpe in the same column and
        # then reason about the comparison. That happened on the first real
        # run: a 0.65 Sharpe was tabulated beside a "deflated Sharpe" of 0.73,
        # which is impossible for a deflated ratio and perfectly ordinary for
        # a probability.
        "prob_sharpe_above_zero": round(psr, 4),
        "prob_beats_best_of_n_trials": round(dsr, 4),
        "expected_max_sharpe_from_luck": round(emax, 6),
        "min_track_record_length_days": round(trl, 1),
        "n_trials_used": used,
        "n_trials_observed": observed,
        "trial_variance_assumption": TRIAL_VARIANCE,
        "units_note": (
            "prob_* fields are probabilities in [0,1], NOT Sharpe ratios. "
            "prob_beats_best_of_n_trials is the deflated Sharpe: the probability "
            "this strategy's true Sharpe beats what the best of n_trials worthless "
            "strategies would have printed. Below 0.5 means the evidence does not "
            "survive the number of things tried. Never tabulate these beside a "
            "Sharpe ratio as though they shared units."
        ),
        "n_trials_note": (
            f"n_trials_used={used} is the count AS OF THIS CALL. Running further "
            "backtests afterwards raises the honest count and makes this figure "
            "too generous. Call analyze_results again on every backtest after the "
            "last one has run, and report those numbers."
        ),
    }
    if overridden:
        out["n_trials_overridden"] = True
        out["note"] = (
            f"n_trials={requested} was below the {observed} backtests actually run "
            "in this session; the session count was used."
        )
    return out
