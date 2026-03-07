import sqlite3
import time
from pathlib import Path
from dataclasses import dataclass


TI_WARN = 0.15
TI_ALERT = 0.30
RE_WARN = 1500
RE_ALERT = 3000
DENSITY_ALERT = 4.0


@dataclass
class AlertEvent:
    frame_id: int
    timestamp: float
    TI: float
    predicted_TI: float
    Re: float
    density_max: float
    shockwave: int
    level: int


class AlertEngine:
    def __init__(self, db_path: str = "db/alerts.db"):
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""CREATE TABLE IF NOT EXISTS alerts(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER,
            timestamp REAL,
            TI REAL,
            predicted_TI REAL,
            Re REAL,
            density_max REAL,
            shockwave INTEGER,
            level INTEGER
        )""")
        self.conn.commit()

    def _log(self, e: AlertEvent):
        self.conn.execute(
            "INSERT INTO alerts VALUES(NULL,?,?,?,?,?,?,?,?)",
            (
                e.frame_id,
                e.timestamp,
                e.TI,
                e.predicted_TI,
                e.Re,
                e.density_max,
                e.shockwave,
                e.level,
            ),
        )

        self.conn.commit()

    def evaluate(self, frame_id: int, metrics: dict, predicted_TI: float) -> AlertEvent:
        TI = metrics["TI"]
        Re = metrics["Re"]
        density = metrics["density_max"]
        shock = metrics["shockwave_flag"]
        if predicted_TI > TI_ALERT or Re > RE_ALERT or density > DENSITY_ALERT:
            level = 2
        elif TI > TI_WARN or Re > RE_WARN:
            level = 1
        else:
            level = 0
        event = AlertEvent(
            frame_id, time.time(), TI, predicted_TI, Re, density, shock, level
        )
        if level > 0:
            self._log(event)
        return event

    def get_log(self) -> list[dict]:
        cur = self.conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
