"""Tests for the tool boundary. Delivered RED.

These are the tests for the file that decides whether "the LLM decides, never
computes" is true. They are grouped by which of the four rules in the module
docstring they defend, and the ones that matter most are:

    test_unknown_argument_key_is_rejected_not_ignored
    test_no_tool_result_contains_a_long_sequence
    test_understated_n_trials_is_overridden_by_the_session_count
    test_dispatch_converts_pipeline_failures_into_tool_errors

Every numeric expectation is either hand-derived or an analytic identity. None
was read out of an implementation.
"""
from __future__ import annotations

import datetime as dt
import json

import polars as pl
import pytest

from falsify.agent import tools as T
from falsify.agent.session import MAX_SUMMARY_BYTES, Session, SessionError


# --- fixtures --------------------------------------------------------------


def _panel(n_tickers: int = 12, n_days: int = 400) -> pl.DataFrame:
    """A deterministic price panel. Ticker i drifts at (i - n/2) bps per day.

    Deterministic on purpose: a sampled panel makes a test that fails once a
    month, and a test that fails once a month gets deleted.
    """
    days = pl.date_range(dt.date(2024, 1, 1), dt.date(2026, 12, 31), "1d", eager=True)
    days = days.filter(days.dt.weekday() <= 5)[:n_days]
    rows = []
    for i in range(n_tickers):
        drift = 1.0 + 0.0004 * (i - n_tickers / 2)
        for j, d in enumerate(days):
            rows.append((f"T{i:02d}", d, 100.0 * drift**j))
    return pl.DataFrame(
        rows, schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}, orient="row"
    )


@pytest.fixture
def s() -> Session:
    return Session()


@pytest.fixture
def patched(monkeypatch):
    """Make fetch_data return the synthetic panel instead of hitting the DB."""
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: _panel())
    return _panel()


@pytest.fixture
def with_feature(s, patched):
    """A session already holding a panel and a mom_12_1 feature frame."""
    ph = T.fetch_data(s)["handle"]
    fh = T.compute_feature(s, ph, "mom_12_1")["handle"]
    return s, ph, fh


# --- rule 1: the menu is closed --------------------------------------------


def test_four_tools_exactly():
    assert {t["name"] for t in T.tool_schemas()} == {
        "fetch_data",
        "compute_feature",
        "run_backtest",
        "analyze_results",
    }


def test_every_schema_has_the_required_shape():
    for t in T.tool_schemas():
        assert t["input_schema"]["type"] == "object"
        assert isinstance(t["input_schema"]["properties"], dict)
        assert isinstance(t["input_schema"]["required"], list)


def test_every_property_is_documented():
    """An undocumented parameter is one the model will use wrongly."""
    for t in T.tool_schemas():
        for name, spec in t["input_schema"]["properties"].items():
            assert spec.get("description"), f"{t['name']}.{name} has no description"


def test_required_lists_only_real_properties():
    for t in T.tool_schemas():
        props = set(t["input_schema"]["properties"])
        assert set(t["input_schema"]["required"]) <= props


def test_descriptions_are_substantive():
    """A vague description is the most common cause of agent failure.

    120 characters is not a quality bar, it is a floor that catches
    "Runs a backtest." as the entire description of the tool that decides what
    the project's headline number means.
    """
    for t in T.tool_schemas():
        assert len(t["description"]) >= 120, f"{t['name']} description is too thin"


def test_momentum_lookback_is_stated_somewhere_in_the_schemas():
    """252 days of history is the fact that most changes the model's plan.

    Without it the model runs 12-1 momentum on a short panel, gets a mostly
    empty signal, and reasons about the result as though the sample were whole.
    """
    blob = json.dumps(T.tool_schemas())
    assert "252" in blob


def test_schemas_are_json_serialisable():
    json.dumps(T.tool_schemas())


def test_feature_menu_is_advertised_as_an_enum():
    """The model must be shown the menu, not left to guess feature names."""
    schema = next(t for t in T.tool_schemas() if t["name"] == "compute_feature")
    enum = schema["input_schema"]["properties"]["feature"].get("enum")
    assert set(enum) == set(T.FEATURES)


# --- rule 2: arguments are validated before dispatch -----------------------


def test_unknown_tool_is_rejected(s):
    with pytest.raises(T.ToolError, match="unknown tool"):
        T.dispatch("delete_database", {}, s)


def test_unknown_tool_error_lists_the_valid_ones(s):
    with pytest.raises(T.ToolError, match="fetch_data"):
        T.dispatch("fetch_datas", {}, s)


def test_unknown_argument_key_is_rejected_not_ignored():
    """Silently dropping an unknown key is the dangerous behaviour.

    A model that sends `n_deciles` when the parameter is `n_buckets` has
    misunderstood something. Ignoring the key runs the backtest at the default
    and returns a result the model then reasons about as though it had chosen
    10 buckets deliberately. The error costs one turn; the silent default
    corrupts the conclusion.
    """
    with pytest.raises(T.ToolError, match="n_deciles"):
        T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "n_deciles": 5})


def test_missing_required_argument_is_rejected():
    with pytest.raises(T.ToolError, match="feature_handle"):
        T.validate_arguments("run_backtest", {"n_buckets": 10})


def test_unknown_tool_is_reported_before_its_arguments():
    """Otherwise the model fixes the arguments of a tool that does not exist."""
    with pytest.raises(T.ToolError, match="unknown tool"):
        T.validate_arguments("run_backtests", {"bogus": 1})


def test_wrong_type_is_rejected():
    with pytest.raises(T.ToolError):
        T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "n_buckets": "ten"})


def test_unknown_feature_is_rejected():
    with pytest.raises(T.ToolError, match="mom_12_1"):
        T.validate_arguments("compute_feature", {"panel_handle": "panel_1", "feature": "alpha"})


@pytest.mark.parametrize("n", [1, 0, -3, 21, 100])
def test_bucket_count_outside_the_range_is_rejected(n):
    with pytest.raises(T.ToolError):
        T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "n_buckets": n})


@pytest.mark.parametrize("n", [2, 5, 10, 20])
def test_bucket_count_inside_the_range_is_accepted(n):
    T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "n_buckets": n})


@pytest.mark.parametrize("bps", [-1.0, 101.0])
def test_absurd_costs_are_rejected(bps):
    with pytest.raises(T.ToolError):
        T.validate_arguments("run_backtest", {"feature_handle": "feature_1", "cost_bps": bps})


def test_unsupported_rebalance_frequency_is_rejected():
    with pytest.raises(T.ToolError, match="monthly"):
        T.validate_arguments(
            "run_backtest", {"feature_handle": "feature_1", "rebalance": "daily"}
        )


def test_valid_arguments_pass_through_unchanged():
    args = {"feature_handle": "feature_1", "n_buckets": 5, "cost_bps": 10.0}
    assert T.validate_arguments("run_backtest", args) == args


def test_optional_arguments_may_be_omitted():
    T.validate_arguments("fetch_data", {})


# --- rule 3: results cross as summaries, never payloads --------------------


def test_fetch_data_returns_a_handle_not_a_panel(s, patched):
    out = T.fetch_data(s)
    assert out["handle"] == "panel_1"
    assert "rows" not in out and "data" not in out


def test_fetch_data_summary_reports_shape(s, patched):
    out = T.fetch_data(s)
    assert out["n_tickers"] == 12
    assert out["n_rows"] == 4800
    assert out["first_date"] == "2024-01-01"


def test_the_panel_is_reachable_through_the_session(s, patched):
    h = T.fetch_data(s)["handle"]
    assert isinstance(s.payload(h, "panel"), pl.DataFrame)


def test_no_tool_result_contains_a_long_sequence(with_feature):
    """The structural check on the no-computation guarantee.

    A tool that returns a series hands the model the raw material to do
    arithmetic on, whatever the docstring claims. Nothing crossing the boundary
    may contain a sequence long enough to be a data series.
    """
    s, ph, fh = with_feature
    results = [
        T.fetch_data(s),
        T.compute_feature(s, ph, "vol_21d"),
        T.run_backtest(s, fh),
    ]
    for out in results:
        for key, value in out.items():
            if isinstance(value, (list, tuple)):
                assert len(value) <= 20, f"{key} carries {len(value)} elements"


def test_every_tool_result_fits_the_summary_cap(with_feature):
    s, ph, fh = with_feature
    for out in (T.fetch_data(s), T.compute_feature(s, ph, "ret_5d"), T.run_backtest(s, fh)):
        assert len(json.dumps(out, default=str)) <= MAX_SUMMARY_BYTES


def test_every_tool_result_is_json_safe(with_feature):
    s, ph, fh = with_feature
    json.dumps(T.run_backtest(s, fh))


# --- rule 4: errors come back as data --------------------------------------


def test_dispatch_converts_pipeline_failures_into_tool_errors(s):
    """A bad handle must surface as ToolError, not UnknownHandle.

    The loop catches one exception type and turns it into an error
    tool_result. Anything else ends the run and discards everything spent on
    it so far.
    """
    with pytest.raises(T.ToolError):
        T.dispatch("compute_feature", {"panel_handle": "panel_9", "feature": "ret_1d"}, s)


def test_dispatch_converts_unexpected_exceptions(s, monkeypatch):
    """dispatch is the last line of defence and must hold for ANY exception.

    The tools catch what they expect. dispatch exists for what they do not: a
    Polars schema error, a division by zero deep in the engine, a typo. Without
    a blanket conversion here, one unanticipated exception ends a run that has
    already been paid for.
    """
    monkeypatch.setattr(T, "fetch_data", lambda *a, **k: 1 / 0)
    with pytest.raises(T.ToolError, match="fetch_data failed"):
        T.dispatch("fetch_data", {}, s)


def test_dispatch_converts_session_errors_raised_from_a_tool(s, monkeypatch):
    """Belt and braces: a Session error escaping a tool is still a ToolError."""
    def boom(*a, **k):
        raise SessionError("no handle 'panel_4' in this run. Available: none yet")

    monkeypatch.setattr(T, "fetch_data", boom)
    with pytest.raises(T.ToolError, match="panel_4"):
        T.dispatch("fetch_data", {}, s)


def test_handle_error_message_reaches_the_model_intact(s, patched):
    """The message is the model's only route to recovery, so it must survive."""
    T.fetch_data(s)
    with pytest.raises(T.ToolError, match="panel_1"):
        T.dispatch("compute_feature", {"panel_handle": "panel_7", "feature": "ret_1d"}, s)


def test_wrong_kind_of_handle_is_a_tool_error(s, patched):
    ph = T.fetch_data(s)["handle"]
    with pytest.raises(T.ToolError):
        T.dispatch("run_backtest", {"feature_handle": ph}, s)


def test_empty_panel_is_an_explicit_error_not_an_empty_result(s, monkeypatch):
    """An empty panel silently produces an empty backtest and a confident
    conclusion about nothing. The model has to be told."""
    monkeypatch.setattr(
        T, "load_panel", lambda *a, **k: pl.DataFrame(
            schema={"ticker": pl.Utf8, "ts": pl.Date, "close": pl.Float64}
        )
    )
    with pytest.raises(T.ToolError, match="no"):
        T.fetch_data(s)


def test_feature_with_no_values_is_an_error(s, monkeypatch):
    """mom_12_1 on a 60-day panel scores nothing. Say so, do not return zero."""
    monkeypatch.setattr(T, "load_panel", lambda *a, **k: _panel(n_days=60))
    ph = T.fetch_data(s)["handle"]
    with pytest.raises(T.ToolError):
        T.compute_feature(s, ph, "mom_12_1")


# --- the tools themselves --------------------------------------------------


def test_compute_feature_reports_coverage_not_just_success(s, patched):
    """Coverage is how the model learns the first year of mom_12_1 is null."""
    ph = T.fetch_data(s)["handle"]
    out = T.compute_feature(s, ph, "mom_12_1")
    assert out["n_non_null"] < out["n_rows"]
    assert out["n_non_null"] == 12 * (400 - 252)


def test_compute_feature_chains_onto_the_stored_panel(s, patched):
    ph = T.fetch_data(s)["handle"]
    fh = T.compute_feature(s, ph, "vol_21d")["handle"]
    assert "vol_21d" in s.payload(fh, "feature").columns


def test_run_backtest_reports_the_invested_window_not_the_full_sample(with_feature):
    """252 days have no positions. Averaging over them understates everything.

    The identity that makes this discriminating: `n_days` comes out of
    `metrics.summary`, `n_invested_days` is counted from the weights. If the
    metrics were computed over the full sample they would disagree, and every
    figure the agent reports would be diluted by a year of flat days while
    still looking entirely plausible.
    """
    s, ph, fh = with_feature
    out = T.run_backtest(s, fh)
    assert out["n_invested_days"] < 400
    assert out["n_days"] == out["n_invested_days"]


def test_run_backtest_summary_carries_the_metrics(with_feature):
    s, ph, fh = with_feature
    out = T.run_backtest(s, fh)
    for k in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        assert k in out


def test_run_backtest_increments_the_trial_count(with_feature):
    s, ph, fh = with_feature
    assert s.n_backtests == 0
    T.run_backtest(s, fh)
    T.run_backtest(s, fh, n_buckets=5)
    assert s.n_backtests == 2


# --- analyze_results, and the anti-cheat -----------------------------------


def test_analyze_results_returns_the_honesty_table(with_feature):
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    out = T.analyze_results(s, bh)
    for k in ("sharpe_per_period", "prob_sharpe_above_zero",
              "expected_max_sharpe_from_luck", "prob_beats_best_of_n_trials",
              "min_track_record_length_days", "n_trials_used"):
        assert k in out


def test_probabilities_are_named_as_probabilities(with_feature):
    """Field names carry the units, because a name is all the reader gets.

    On the first real run a key called `deflated_sharpe` was tabulated by the
    model in the same column as an actual Sharpe: 0.65 raw beside 0.73
    "deflated", which is impossible for a deflated ratio and perfectly ordinary
    for a probability. The numbers were right and the interpretation was wrong,
    and the name caused it.
    """
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    out = T.analyze_results(s, bh)
    for k in ("prob_sharpe_above_zero", "prob_beats_best_of_n_trials"):
        assert 0.0 <= out[k] <= 1.0
    assert "deflated_sharpe" not in out, "a bare 'deflated_sharpe' key invites the unit error"
    assert "probabilit" in out["units_note"].lower()


def test_the_output_warns_that_the_trial_count_is_as_of_now(with_feature):
    """The anti-cheat stops understating; it cannot stop analysing too early.

    An agent that analyses backtest_1 and then runs two more has a first row
    deflated for one trial when the honest count turned out to be three. The
    guard cannot see the future, so the output says so and the runner
    recomputes every row at the final count.
    """
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    note = T.analyze_results(s, bh)["n_trials_note"].lower()
    assert "as of this call" in note


def test_n_trials_defaults_to_what_the_session_observed(with_feature):
    s, ph, fh = with_feature
    for n in (10, 5, 3):
        T.run_backtest(s, fh, n_buckets=n)
    bh = s.handles("backtest")[-1]
    assert T.analyze_results(s, bh)["n_trials_used"] == 3


def test_understated_n_trials_is_overridden_by_the_session_count(with_feature):
    """THE ANTI-CHEAT, and the reason this file is hand-built.

    The model has run nine backtests and reports one. Understating N is how a
    deflated Sharpe gets quietly re-inflated, and a model summarising its own
    work has every incentive to forget the variants it abandoned. The Session's
    count wins, and the substitution is reported rather than made silently.
    """
    s, ph, fh = with_feature
    for n in range(2, 11):
        T.run_backtest(s, fh, n_buckets=n)
    bh = s.handles("backtest")[-1]
    out = T.analyze_results(s, bh, n_trials=1)
    assert out["n_trials_used"] == 9
    assert out["n_trials_observed"] == 9
    assert out.get("n_trials_overridden") is True


def test_overstated_n_trials_is_accepted(with_feature):
    """More deflation is the conservative direction, and the model may know
    about trials from outside this run."""
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    out = T.analyze_results(s, bh, n_trials=50)
    assert out["n_trials_used"] == 50
    assert not out.get("n_trials_overridden")


def test_more_trials_never_raises_the_deflated_sharpe(with_feature):
    """Analytic identity: E[max SR] rises with N, so DSR must not rise.

    If this fails, the deflation is wired backwards and every verdict the agent
    produces is flattering.
    """
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    few = T.analyze_results(s, bh, n_trials=2)["prob_beats_best_of_n_trials"]
    many = T.analyze_results(s, bh, n_trials=100)["prob_beats_best_of_n_trials"]
    assert many <= few


def test_analyze_results_does_not_count_as_a_trial(with_feature):
    """Analysing a result twice is not two strategies tried."""
    s, ph, fh = with_feature
    bh = T.run_backtest(s, fh)["handle"]
    T.analyze_results(s, bh)
    T.analyze_results(s, bh)
    assert s.n_backtests == 1


def test_dispatch_routes_every_tool(s, patched):
    """End to end through dispatch, the way the loop will call it."""
    ph = T.dispatch("fetch_data", {}, s)["handle"]
    fh = T.dispatch("compute_feature", {"panel_handle": ph, "feature": "mom_12_1"}, s)["handle"]
    bh = T.dispatch("run_backtest", {"feature_handle": fh, "n_buckets": 5}, s)["handle"]
    out = T.dispatch("analyze_results", {"backtest_handle": bh}, s)
    assert out["n_trials_used"] == 1
