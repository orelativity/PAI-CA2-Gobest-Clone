import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "gobest_history.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS driver_history (
            driver_id INTEGER PRIMARY KEY,
            total_trips INTEGER,
            dangerous_trips INTEGER,
            dangerous_rate REAL,
            avg_harsh_accel REAL,
            last_updated TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS trip_predictions (
            bookingID INTEGER,
            driver_id INTEGER,
            pred_proba REAL,
            pred_label INTEGER,
            threshold REAL,
            created_at TEXT
        )
        """)

        conn.commit()


def fetch_db_stats():
    """
    returns: dict with counts for display
    """
    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM trip_predictions")
        total_preds = int(cur.fetchone()[0])

        cur.execute("SELECT COUNT(DISTINCT driver_id) FROM driver_history")
        total_drivers = int(cur.fetchone()[0])

        cur.execute("SELECT COUNT(*) FROM driver_history WHERE dangerous_rate >= 0.5")
        high_risk_drivers = int(cur.fetchone()[0])

        return {
            "total_preds": total_preds,
            "total_drivers": total_drivers,
            "high_risk_drivers": high_risk_drivers,
        }


def fetch_recent_predictions(limit=12):
    """
    returns list of tuples:
    (created_at, bookingID, driver_id, pred_proba, pred_label, threshold)
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT created_at, bookingID, driver_id, pred_proba, pred_label, threshold
        FROM trip_predictions
        ORDER BY created_at DESC
        LIMIT ?
        """, (int(limit),))
        return cur.fetchall()


def fetch_top_drivers(limit=10):
    """
    returns list of tuples:
    (driver_id, total_trips, dangerous_trips, dangerous_rate, avg_harsh_accel, last_updated)
    sorted by dangerous_rate desc, then total_trips desc
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT driver_id, total_trips, dangerous_trips, dangerous_rate, avg_harsh_accel, last_updated
        FROM driver_history
        ORDER BY dangerous_rate DESC, total_trips DESC
        LIMIT ?
        """, (int(limit),))
        return cur.fetchall()


def fetch_driver_history(driver_id):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
        SELECT total_trips, dangerous_trips, dangerous_rate, avg_harsh_accel, last_updated
        FROM driver_history
        WHERE driver_id = ?
        """, (int(driver_id),))
        return cur.fetchone()


def save_predictions(preds_df, threshold):
    """
    saves every row in preds_df into trip_predictions
    expects columns: bookingID, driver_id, pred_proba, pred_label
    """
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        cur = conn.cursor()
        for _, r in preds_df.iterrows():
            cur.execute("""
            INSERT INTO trip_predictions
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                int(r["bookingID"]),
                int(r["driver_id"]),
                float(r["pred_proba"]),
                int(r["pred_label"]),
                float(threshold),
                now,
            ))
        conn.commit()


def update_driver_history(preds_df):
    """
    aggregates the current batch prediction output into driver_history
    """
    now = datetime.utcnow().isoformat()

    with get_conn() as conn:
        cur = conn.cursor()

        for driver_id, g in preds_df.groupby("driver_id"):
            total = int(len(g))
            dangerous = int((g["pred_label"] == 1).sum())
            rate = dangerous / total if total else 0.0

            # we assume harsh_acceleration_count exists in engineered features
            avg_harsh = float(g["harsh_acceleration_count"].mean()) if "harsh_acceleration_count" in g.columns else 0.0

            cur.execute("""
            INSERT INTO driver_history
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(driver_id) DO UPDATE SET
                total_trips = total_trips + excluded.total_trips,
                dangerous_trips = dangerous_trips + excluded.dangerous_trips,
                dangerous_rate =
                    CAST(dangerous_trips + excluded.dangerous_trips AS REAL) /
                    CAST(total_trips + excluded.total_trips AS REAL),
                avg_harsh_accel = excluded.avg_harsh_accel,
                last_updated = excluded.last_updated
            """, (
                int(driver_id),
                total,
                dangerous,
                float(rate),
                float(avg_harsh),
                now,
            ))

        conn.commit()


def reset_db():
    """
    convenience for demos/testing
    """
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM trip_predictions")
        cur.execute("DELETE FROM driver_history")
        conn.commit()
