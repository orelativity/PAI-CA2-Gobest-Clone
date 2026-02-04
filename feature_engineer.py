import numpy as np
import pandas as pd


THRESH = {
    "harsh_accel": 4.5,
    "harsh_brake": -5.5,
    "sharp_turn_gyro_z": 2.0,
    "speeding": 33.3,       # ~120 km/h
    "high_speed": 25.0,
    "gps_bad": 30.0,
    "w5": 5,
    "w10": 10,
}


def _coerce_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return 1
    if s in {"false", "0", "no", "n"}:
        return 0
    try:
        return int(float(s))
    except Exception:
        return np.nan


def engineer_features_from_raw_tables(sensor_df: pd.DataFrame, driver_df: pd.DataFrame, safety_df: pd.DataFrame) -> pd.DataFrame:
    """
    converts raw tables to one row per bookingID (trip-level features)

    inputs required:
    - sensor_df must include bookingID + sensor columns (speed, accel, gyro, second, accuracy optional)
    - safety_df must include bookingID + driver_id (label optional)
    - driver_df used for extra metadata if needed (optional for xgboost in your pipeline)
    """
    sensor_df = sensor_df.copy()
    driver_df = driver_df.copy()
    safety_df = safety_df.copy()

    sensor_df.columns = [c.strip() for c in sensor_df.columns]
    driver_df.columns = [c.strip() for c in driver_df.columns]
    safety_df.columns = [c.strip() for c in safety_df.columns]

    if "bookingID" not in sensor_df.columns:
        # most common: pandas wrote index as "Unnamed: 0"
        if "Unnamed: 0" in sensor_df.columns:
            sensor_df = sensor_df.rename(columns={"Unnamed: 0": "bookingID"})
        else:
            # fallback: treat first column as bookingID if it matches safety_labels bookingID
            first_col = sensor_df.columns[0]
            if first_col != "bookingID":
                # check overlap with safety bookingIDs
                try:
                    cand = pd.to_numeric(sensor_df[first_col], errors="coerce")
                    safety_ids = set(pd.to_numeric(safety_df["bookingID"], errors="coerce").dropna().astype(int).tolist())
                    overlap = int(cand.dropna().astype(int).isin(safety_ids).mean() * 100)

                    # if at least 50% overlap, assume this column is bookingID
                    if overlap >= 50:
                        sensor_df = sensor_df.rename(columns={first_col: "bookingID"})
                except Exception:
                    pass

        if "bookingID" not in sensor_df.columns:
            raise ValueError("sensor_data must include bookingID column")

        # always coerce bookingID to numeric (critical)
        sensor_df["bookingID"] = pd.to_numeric(sensor_df["bookingID"], errors="coerce")


    if "bookingID" not in safety_df.columns or "driver_id" not in safety_df.columns:
        raise ValueError("safety_labels must include bookingID and driver_id columns")

    if "label" in safety_df.columns:
        safety_df["label"] = safety_df["label"].apply(_coerce_label)

    # numeric coercion
    for c in ["second", "speed", "accuracy", "bearing",
              "acceleration_x", "acceleration_y", "acceleration_z",
              "gyro_x", "gyro_y", "gyro_z"]:
        if c in sensor_df.columns:
            sensor_df[c] = pd.to_numeric(sensor_df[c], errors="coerce").fillna(0.0)
        else:
            sensor_df[c] = 0.0

    sensor_df = sensor_df.sort_values(["bookingID", "second"])

    rows = []
    for bid, g in sensor_df.groupby("bookingID"):
        n = len(g)

        # duration (simple)
        trip_duration_sec = float(max(0.0, g["second"].max() - g["second"].min())) if n else 0.0

        # distance estimate (sum speed over seconds)
        total_distance_km = float(g["speed"].sum() / 1000.0)

        # magnitudes
        ax, ay, az = g["acceleration_x"].to_numpy(), g["acceleration_y"].to_numpy(), g["acceleration_z"].to_numpy()
        gx, gy, gz = g["gyro_x"].to_numpy(), g["gyro_y"].to_numpy(), g["gyro_z"].to_numpy()

        accel_mag = np.sqrt(ax*ax + ay*ay + az*az)
        gyro_mag = np.sqrt(gx*gx + gy*gy + gz*gz)

        # event counts
        harsh_acceleration_count = int(np.sum(g["acceleration_x"] > THRESH["harsh_accel"]))
        harsh_braking_count = int(np.sum(g["acceleration_x"] < THRESH["harsh_brake"]))
        sharp_turn_count = int(np.sum(np.abs(g["gyro_z"]) > THRESH["sharp_turn_gyro_z"]))
        speeding_event_count = int(np.sum(g["speed"] > THRESH["speeding"]))
        phone_distraction_count = int(np.sum(g["accuracy"] > THRESH["gps_bad"]))

        # rolling signals
        speed_rolling_std_5s = float(pd.Series(g["speed"]).rolling(THRESH["w5"], min_periods=1).std().mean())
        accel_x_rolling_max_10s = float(pd.Series(g["acceleration_x"]).rolling(THRESH["w10"], min_periods=1).max().mean())
        gyro_z_rolling_range_5s = float(
            (pd.Series(g["gyro_z"]).rolling(THRESH["w5"], min_periods=1).max() -
             pd.Series(g["gyro_z"]).rolling(THRESH["w5"], min_periods=1).min()).mean()
        )

        # speed change rate
        speed_change_rate = float(np.mean(np.abs(np.diff(g["speed"].to_numpy())))) if n > 1 else 0.0

        # jerk
        jerk_x = np.diff(ax, prepend=ax[0] if n else 0.0)
        jerk_y = np.diff(ay, prepend=ay[0] if n else 0.0)
        jerk_z = np.diff(az, prepend=az[0] if n else 0.0)
        jerk_mag = np.sqrt(jerk_x*jerk_x + jerk_y*jerk_y + jerk_z*jerk_z)

        # ratios
        harsh_decel_at_high_speed_count = int(np.sum((g["acceleration_x"] < THRESH["harsh_brake"]) & (g["speed"] > THRESH["high_speed"])))
        accel_variance_normalized_by_speed = float(np.var(accel_mag) / (np.mean(g["speed"]) + 1e-6))

        row = {
            "bookingID": bid,
            "trip_duration_sec": trip_duration_sec,
            "total_distance_km": total_distance_km,
            "avg_gps_accuracy": float(g["accuracy"].mean()),
            "harsh_acceleration_count": harsh_acceleration_count,
            "harsh_braking_count": harsh_braking_count,
            "sharp_turn_count": sharp_turn_count,
            "speeding_event_count": speeding_event_count,
            "phone_distraction_count": phone_distraction_count,
            "avg_acceleration_magnitude": float(np.mean(accel_mag)) if n else 0.0,
            "max_acceleration_magnitude": float(np.max(accel_mag)) if n else 0.0,
            "speed_rolling_std_5s": speed_rolling_std_5s,
            "accel_x_rolling_max_10s": accel_x_rolling_max_10s,
            "gyro_z_rolling_range_5s": gyro_z_rolling_range_5s,
            "speed_change_rate": speed_change_rate,
            "gyro_total_rotation": float(np.sum(np.abs(gx)) + np.sum(np.abs(gy)) + np.sum(np.abs(gz))),
            "gyro_magnitude_max": float(np.max(gyro_mag)) if n else 0.0,
            "gyro_z_peak_count": int(np.sum(np.abs(gz) > 1.5)),
            "gyro_stability_ratio": float(np.mean((np.abs(gx) < 0.5) & (np.abs(gy) < 0.5) & (np.abs(gz) < 0.5))),
            "speed_accel_product": float(np.mean(g["speed"].to_numpy() * accel_mag)) if n else 0.0,
            "harsh_decel_at_high_speed_count": harsh_decel_at_high_speed_count,
            "accel_variance_normalized_by_speed": accel_variance_normalized_by_speed,
            "jerk_x_mean": float(np.mean(jerk_x)) if n else 0.0,
            "jerk_y_max": float(np.max(jerk_y)) if n else 0.0,
            "jerk_z_std": float(np.std(jerk_z)) if n else 0.0,
            "jerk_magnitude_std": float(np.std(jerk_mag)) if n else 0.0,
        }

        rows.append(row)

    engineered = pd.DataFrame(rows)

    # attach driver_id + label from safety
    meta = safety_df[["bookingID", "driver_id"] + (["label"] if "label" in safety_df.columns else [])].drop_duplicates("bookingID")
    engineered = engineered.merge(meta, on="bookingID", how="left")

    # fill missing driver_id with -1 to avoid crashes (but caller should validate)
    engineered["driver_id"] = pd.to_numeric(engineered["driver_id"], errors="coerce").fillna(-1).astype(int)

    # ensure label exists if not provided
    if "label" not in engineered.columns:
        engineered["label"] = np.nan

    return engineered
