"""
Local SQLite state of the tool. Phase 2 stores the asset pool and universe
snapshots; later phases add orders, fills and equity.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool_assets (
    symbol     TEXT PRIMARY KEY,
    name       TEXT,
    exchange   TEXT,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS universe_snapshots (
    as_of            TEXT NOT NULL,   -- session whose close the screen used
    effective        TEXT NOT NULL,   -- first session the snapshot applies to
    rank             INTEGER NOT NULL,
    symbol           TEXT NOT NULL,
    median_dollar_volume REAL,
    close            REAL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (as_of, symbol)
);
"""


class StateDB:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- asset pool ----------

    def replace_pool(self, assets: pd.DataFrame) -> None:
        now = pd.Timestamp.now(tz="UTC").isoformat()
        rows = [(r.symbol, r.name, r.exchange, now) for r in assets.itertuples()]
        with self._conn() as c:
            c.execute("DELETE FROM pool_assets")
            c.executemany("INSERT INTO pool_assets VALUES (?, ?, ?, ?)", rows)

    def pool(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM pool_assets ORDER BY symbol", c)

    def pool_symbols(self) -> List[str]:
        return self.pool()["symbol"].tolist()

    # ---------- universe snapshots ----------

    def save_snapshot(self, as_of: pd.Timestamp, effective: pd.Timestamp, table: pd.DataFrame) -> None:
        """`table`: index symbol, columns rank, median_dollar_volume, close."""
        now = pd.Timestamp.now(tz="UTC").isoformat()
        a, e = pd.Timestamp(as_of).date().isoformat(), pd.Timestamp(effective).date().isoformat()
        rows = [(a, e, int(r["rank"]), sym, float(r["median_dollar_volume"]), float(r["close"]), now)
                for sym, r in table.iterrows()]
        with self._conn() as c:
            c.execute("DELETE FROM universe_snapshots WHERE as_of = ?", (a,))
            c.executemany("INSERT INTO universe_snapshots VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def latest_snapshot(self, before: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
        """Most recent snapshot, or the most recent one strictly before `before`."""
        limit = pd.Timestamp(before).date().isoformat() if before is not None else "9999-12-31"
        with self._conn() as c:
            df = pd.read_sql(
                "SELECT * FROM universe_snapshots WHERE as_of = "
                "(SELECT MAX(as_of) FROM universe_snapshots WHERE as_of < ?) ORDER BY rank",
                c, params=(limit,))
        return None if df.empty else df
