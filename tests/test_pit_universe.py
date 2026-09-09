"""Point-in-time universe reconstruction.

No network, no git, no database: every fixture is hand-written CSV text, so a
failure here is a logic failure and never a connectivity one.
"""
from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from falsify.data.pit_universe import (
    dropped_tickers,
    members_as_of,
    membership_panel,
    normalise_ticker,
    parse_constituents,
    parse_snapshots,
)

D = dt.date


def _csv(tickers: list[str], header: str = "Symbol,Security,GICS Sector") -> str:
    """Constituents CSV text with the given symbols."""
    lines = [header] + [f"{t},Some Company {t},Tech" for t in tickers]
    return "\n".join(lines) + "\n"


def _pad(n: int, prefix: str = "T") -> list[str]:
    """n filler tickers clearing the MIN_CONSTITUENTS floor. Five capitals max,
    so the pattern accepts them: TAAA, TAAB, ..."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = []
    for i in range(n):
        a, b, c = i // 676, (i // 26) % 26, i % 26
        out.append(f"{prefix}{letters[a]}{letters[b]}{letters[c]}")
    return out


# --------------------------------------------------------------------------
# normalise_ticker: the three real quirks in this source's history
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AAPL", "AAPL"),
        ("  msft  ", "MSFT"),                  # whitespace and case
        ("BRK-B", "BRK.B"),                    # hyphen class share -> dot
        ("BF-B", "BF.B"),
        ("RVTY (Previously PKI)", "RVTY"),     # annotated rename
        ("American Airlines Group", None),     # column-shift artifact
        ("", None),
        ("   ", None),
        ("TOOLONGTICKER", None),
    ],
)
def test_normalise_ticker(raw, expected):
    assert normalise_ticker(raw) == expected


def test_normalisation_matches_polygon_convention():
    """BRK-B and BRK.B must not survive as two different names.

    daily_bars is keyed by the Polygon string. A snapshot saying BRK-B would
    miss the join silently, and Berkshire would look like a non-member for
    every date drawn from that revision.
    """
    assert normalise_ticker("BRK-B") == normalise_ticker("BRK.B") == "BRK.B"


# --------------------------------------------------------------------------
# parse_constituents: header shape has changed three times in this file's life
# --------------------------------------------------------------------------

def test_parse_handles_every_historical_header():
    for header in (
        "Symbol,Security,GICS Sector,GICS Sub-Industry",
        "Symbol,Company,GICS Sector",
        "Symbol,Name,Sector",
    ):
        assert parse_constituents(_csv(["AAPL", "MSFT"], header)) == {"AAPL", "MSFT"}


def test_parse_rejects_file_without_symbol_column():
    """A reshuffle that renamed the symbol column must fail loudly-empty. A
    partial or positionally-guessed set would poison the audit with a snapshot
    that looks plausible and is wrong."""
    assert parse_constituents("Ticker,Name\nAAPL,Apple\n") == set()


def test_parse_drops_junk_rows_but_keeps_the_rest():
    text = _csv(["AAPL", "American Airlines Group", "MSFT"])
    assert parse_constituents(text) == {"AAPL", "MSFT"}


def test_parse_deduplicates():
    assert parse_constituents(_csv(["AAPL", "AAPL", "MSFT"])) == {"AAPL", "MSFT"}


# --------------------------------------------------------------------------
# parse_snapshots
# --------------------------------------------------------------------------

def test_parse_snapshots_shape_and_sorting():
    base = _pad(400)
    snaps = parse_snapshots(
        [
            (D(2025, 3, 1), _csv(base + ["AAPL"])),
            (D(2025, 1, 1), _csv(base + ["MSFT"])),
        ]
    )
    assert snaps.columns == ["ticker", "index_name", "as_of"]
    assert snaps["as_of"].to_list() == sorted(snaps["as_of"].to_list())
    assert snaps.filter(pl.col("as_of") == D(2025, 1, 1)).height == 401
    assert set(snaps["index_name"].unique()) == {"SP500"}


def test_short_revision_is_discarded():
    """A truncated or half-written revision must not become a snapshot: it
    would read as ~450 companies leaving the index on a single day, every one
    scored as a survivorship casualty."""
    snaps = parse_snapshots(
        [
            (D(2025, 1, 1), _csv(_pad(400))),
            (D(2025, 2, 1), _csv(["AAPL", "MSFT"])),   # stub commit
        ]
    )
    assert snaps["as_of"].unique().to_list() == [D(2025, 1, 1)]


def test_same_day_commits_collapse_to_one_snapshot():
    """universe_snapshot is keyed by (ticker, index_name, as_of), so two
    snapshots sharing a date are indistinguishable in the table. Collapse them
    here rather than hitting a primary-key collision on insert."""
    base = _pad(400)
    snaps = parse_snapshots(
        [
            (D(2025, 1, 1), _csv(base + ["AAPL"])),
            (D(2025, 1, 1), _csv(base + ["MSFT"])),
        ]
    )
    assert snaps.filter(pl.col("as_of") == D(2025, 1, 1)).height == 401


def test_empty_input_returns_typed_empty_frame():
    snaps = parse_snapshots([])
    assert snaps.is_empty()
    assert snaps.schema["as_of"] == pl.Date


# --------------------------------------------------------------------------
# members_as_of: the lookahead firewall for membership
# --------------------------------------------------------------------------

@pytest.fixture
def three_snapshots() -> pl.DataFrame:
    """XOM is dropped after Jan; NVDA is added in Mar."""
    base = _pad(400)
    return parse_snapshots(
        [
            (D(2025, 1, 15), _csv(base + ["AAPL", "XOM"])),
            (D(2025, 2, 15), _csv(base + ["AAPL"])),
            (D(2025, 3, 15), _csv(base + ["AAPL", "NVDA"])),
        ]
    )


def test_members_as_of_uses_the_latest_snapshot_at_or_before(three_snapshots):
    assert "XOM" in members_as_of(three_snapshots, D(2025, 1, 15))   # on the date
    assert "XOM" in members_as_of(three_snapshots, D(2025, 2, 1))    # between
    assert "XOM" not in members_as_of(three_snapshots, D(2025, 2, 15))
    assert "XOM" not in members_as_of(three_snapshots, D(2025, 6, 1))


def test_members_as_of_never_looks_forward(three_snapshots):
    """NVDA joins in March and must be invisible in January.

    The failure the whole module exists to prevent. A future addition leaking
    into a past date is survivorship bias with extra steps: the backtest holds
    names selected because they were later successful enough to be added.
    """
    assert "NVDA" not in members_as_of(three_snapshots, D(2025, 1, 20))
    assert "NVDA" not in members_as_of(three_snapshots, D(2025, 3, 14))
    assert "NVDA" in members_as_of(three_snapshots, D(2025, 3, 15))


def test_date_before_first_snapshot_is_empty_not_earliest(three_snapshots):
    """Falling back to the earliest snapshot would import 2025 membership
    into 2024. Empty is the honest answer: we do not know."""
    assert members_as_of(three_snapshots, D(2024, 12, 31)) == set()


def test_members_as_of_empty_snapshots():
    assert members_as_of(pl.DataFrame(schema={"ticker": pl.Utf8, "index_name": pl.Utf8,
                                              "as_of": pl.Date}), D(2025, 1, 1)) == set()


# --------------------------------------------------------------------------
# membership_panel
# --------------------------------------------------------------------------

def test_membership_panel_agrees_with_members_as_of(three_snapshots):
    """Two independent implementations of the same question must agree.
    membership_panel is a vectorised as-of join; members_as_of is a filter and
    a max. Disagreement means the join is wrong."""
    dates = pl.Series("ts", [D(2025, 1, 20), D(2025, 2, 20), D(2025, 3, 20)])
    panel = membership_panel(three_snapshots, dates)
    for d in dates:
        from_panel = set(panel.filter(pl.col("ts") == d)["ticker"].to_list())
        assert from_panel == members_as_of(three_snapshots, d)


def test_membership_panel_excludes_dates_before_first_snapshot(three_snapshots):
    dates = pl.Series("ts", [D(2024, 6, 1), D(2025, 1, 20)])
    panel = membership_panel(three_snapshots, dates)
    assert panel.filter(pl.col("ts") == D(2024, 6, 1)).is_empty()
    assert not panel.filter(pl.col("ts") == D(2025, 1, 20)).is_empty()


def test_membership_panel_shows_an_exit(three_snapshots):
    """A name that leaves must stop appearing, not persist by carry-forward.

    The as-of join carries the LAST snapshot forward. Were a snapshot
    represented only by the tickers present in it, XOM's absence in February
    would leave January's row as the most recent XOM record, held forever.
    """
    dates = pl.Series("ts", [D(2025, 1, 20), D(2025, 2, 20)])
    panel = membership_panel(three_snapshots, dates)
    jan = set(panel.filter(pl.col("ts") == D(2025, 1, 20))["ticker"].to_list())
    feb = set(panel.filter(pl.col("ts") == D(2025, 2, 20))["ticker"].to_list())
    assert "XOM" in jan
    assert "XOM" not in feb


def test_membership_panel_empty_snapshots_is_typed_empty():
    empty = pl.DataFrame(schema={"ticker": pl.Utf8, "index_name": pl.Utf8, "as_of": pl.Date})
    panel = membership_panel(empty, pl.Series("ts", [D(2025, 1, 1)]))
    assert panel.is_empty()
    assert panel.schema["ts"] == pl.Date


# --------------------------------------------------------------------------
# dropped_tickers: the ingest list for the survivorship audit
# --------------------------------------------------------------------------

def test_dropped_tickers_finds_exits_only(three_snapshots):
    dropped = dropped_tickers(three_snapshots)
    assert "XOM" in dropped          # was a member, is not now
    assert "AAPL" not in dropped     # member throughout
    assert "NVDA" not in dropped     # added, still a member


def test_dropped_tickers_respects_the_since_window(three_snapshots):
    """A name that left before the backtest window is not a casualty OF that
    window and would waste an ingest call."""
    assert "XOM" not in dropped_tickers(three_snapshots, since=D(2025, 2, 15))
    assert "XOM" in dropped_tickers(three_snapshots, since=D(2025, 1, 15))
