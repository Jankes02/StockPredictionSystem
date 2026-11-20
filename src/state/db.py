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
-- live (paper) daily cycle
CREATE TABLE IF NOT EXISTS live_runs (
    run_id       TEXT PRIMARY KEY,   -- UTC timestamp of the run
    session      TEXT NOT NULL,      -- session whose open the orders target
    as_of        TEXT NOT NULL,      -- last close the decisions used
    mode         TEXT NOT NULL,      -- dry | live
    status       TEXT NOT NULL,      -- planned | submitted | skipped | halted | failed
    equity       REAL,
    cash         REAL,
    target_gross REAL,
    n_orders     INTEGER,
    message      TEXT
);
CREATE TABLE IF NOT EXISTS live_targets (
    run_id  TEXT NOT NULL,
    symbol  TEXT NOT NULL,
    weight  REAL NOT NULL,
    price   REAL,
    PRIMARY KEY (run_id, symbol)
);
CREATE TABLE IF NOT EXISTS live_orders (
    client_order_id  TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    session          TEXT NOT NULL,
    symbol           TEXT NOT NULL,
    side             TEXT NOT NULL,
    qty              REAL NOT NULL,
    est_price        REAL,
    reason           TEXT,
    status           TEXT NOT NULL,
    broker_id        TEXT,
    filled_qty       REAL,
    filled_avg_price REAL,
    updated_at       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_equity (
    date        TEXT PRIMARY KEY,    -- broker equity observed before the session's open
    equity      REAL NOT NULL,
    cash        REAL,
    recorded_at TEXT NOT NULL
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

    # ---------- live daily cycle ----------

    @staticmethod
    def _now() -> str:
        return pd.Timestamp.now(tz="UTC").isoformat()

    def record_run(self, run_id: str, session, as_of, mode: str, status: str, equity=None, cash=None,
                   target_gross=None, n_orders=None, message: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO live_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (run_id, pd.Timestamp(session).date().isoformat(), pd.Timestamp(as_of).date().isoformat(),
                       mode, status, equity, cash, target_gross, n_orders, message))

    def record_targets(self, run_id: str, weights: dict, prices: dict) -> None:
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO live_targets VALUES (?, ?, ?, ?)",
                          [(run_id, s, float(w), prices.get(s)) for s, w in weights.items()])

    def record_order(self, run_id: str, session, client_order_id: str, symbol: str, side: str, qty: float,
                     est_price: float, reason: str, status: str, broker_id: str = "") -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO live_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      (client_order_id, run_id, pd.Timestamp(session).date().isoformat(), symbol, side,
                       float(qty), est_price, reason, status, broker_id, None, None, self._now()))

    def update_order_fill(self, client_order_id: str, status: str, filled_qty: float,
                          filled_avg_price) -> None:
        with self._conn() as c:
            c.execute("UPDATE live_orders SET status = ?, filled_qty = ?, filled_avg_price = ?, updated_at = ? "
                      "WHERE client_order_id = ?",
                      (status, filled_qty, filled_avg_price, self._now(), client_order_id))

    def live_orders(self, session=None) -> pd.DataFrame:
        with self._conn() as c:
            if session is None:
                return pd.read_sql("SELECT * FROM live_orders ORDER BY session, symbol", c)
            return pd.read_sql("SELECT * FROM live_orders WHERE session = ? ORDER BY symbol", c,
                               params=(pd.Timestamp(session).date().isoformat(),))

    def record_equity(self, date, equity: float, cash: float) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO live_equity VALUES (?, ?, ?, ?)",
                      (pd.Timestamp(date).date().isoformat(), float(equity), float(cash), self._now()))

    def equity_history(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM live_equity ORDER BY date", c)

    def live_runs(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT * FROM live_runs ORDER BY run_id", c)
