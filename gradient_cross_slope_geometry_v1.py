"""
Gradient & Crossfall Prediction (Geometry-based )
-----------------------------------------------------------
- Loads geometry.csv without headers
- Loads JSON accelerometer files
- Matches accel → geometry using BallTree (haversine)
- Trains Random Forest + Dummy baseline
- Saves full metrics + plots + models + per-frame predictions
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
import joblib

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\KarriBhavya\PycharmProjects\Gradient_cross_Slope_geometry\Data_1"
GEOM_PATH = os.path.join(DATA_DIR, "geometry.csv")

JSON_FILES = [
    "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834 (1).json",
    "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834 (2).json",
    "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834.json"

]

OUT_METRICS = "results_rf_metrics_v3"
OUT_MODELS = "models_rf_v3"
OUT_PERFRAME = "results_rf_perframe_v3"
OUT_PLOTS = "plots_rf_v3"

os.makedirs(OUT_METRICS, exist_ok=True)
os.makedirs(OUT_MODELS, exist_ok=True)
os.makedirs(OUT_PERFRAME, exist_ok=True)
os.makedirs(OUT_PLOTS, exist_ok=True)

ROLL_WINDOW_FRAMES = 20
MAX_DISTANCE_METERS = 50


# ---------------- HELPERS ----------------
def load_geometry(path):
    headers = [
        "ID", "networkID", "sectionName", "locFrom", "locTo", "lane",
        "measDate", "gradient", "crossfall", "RAMMID", "latitude", "longitude"
    ]

    df = pd.read_csv(path, header=None, names=headers)

    df["gradient"] = pd.to_numeric(df["gradient"], errors="coerce")
    df["crossfall"] = pd.to_numeric(df["crossfall"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df.dropna(subset=["gradient", "crossfall", "latitude", "longitude"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_json_fast(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for rec in data:
        acc = rec.get("accelerometer")
        if not acc:
            continue

        acc_arr = np.array([[a["x"], a["y"], a["z"]] for a in acc])

        gyro = rec.get("gyroscope", [])
        gyro_arr = np.array([[g["x"], g["y"], g["z"]] for g in gyro]) if gyro else np.zeros((1, 3))

        rows.append({
            "MT": rec.get("MT"),
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "speed": float(rec.get("speed", 0)),
            "ax": acc_arr[:, 0].mean(),
            "ay": acc_arr[:, 1].mean(),
            "az": acc_arr[:, 2].mean(),
            "gz": gyro_arr[:, 2].mean(),
        })

    return pd.DataFrame(rows)


def match_accel_to_geometry_fast(accel_df, geom_df, max_dist):
    accel_coords = np.radians(accel_df[["lat", "lon"]].dropna())
    geom_coords = np.radians(geom_df[["latitude", "longitude"]].dropna())

    tree = BallTree(geom_coords, metric="haversine")
    dist, idx = tree.query(accel_coords, k=1)

    dist_m = dist[:, 0] * 6371000

    matches = []
    for i, d in enumerate(dist_m):
        if d < max_dist:
            g = geom_df.iloc[idx[i][0]]
            a = accel_df.iloc[i]

            matches.append({
                "lat": a["lat"],
                "lon": a["lon"],
                "ax": a["ax"],
                "ay": a["ay"],
                "az": a["az"],
                "gz": a["gz"],
                "speed": a["speed"],
                "gradient": g["gradient"],
                "crossfall": g["crossfall"],
            })

    return pd.DataFrame(matches)


# ---------------- TRAINING (FULL METRICS + PLOTS) ----------------
def train_rf(gt):
    X = gt[["ax", "ay", "az", "gz", "speed"]]
    y_g = gt["gradient"]
    y_c = gt["crossfall"]

    Xtr, Xte, ytr_g, yte_g, ytr_c, yte_c = train_test_split(
        X, y_g, y_c, test_size=0.3, random_state=42
    )

    rf_g = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_c = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_g.fit(Xtr, ytr_g)
    rf_c.fit(Xtr, ytr_c)

    dum_g = DummyRegressor(strategy="mean").fit(Xtr, ytr_g)
    dum_c = DummyRegressor(strategy="mean").fit(Xtr, ytr_c)

    pred_g = rf_g.predict(Xte)
    pred_c = rf_c.predict(Xte)
    pred_g_dummy = dum_g.predict(Xte)
    pred_c_dummy = dum_c.predict(Xte)

    # ---- Full metric calculations ----
    grad_rmse = np.sqrt(mean_squared_error(yte_g, pred_g))
    cross_rmse = np.sqrt(mean_squared_error(yte_c, pred_c))

    grad_drmse = np.sqrt(mean_squared_error(yte_g, pred_g_dummy))
    cross_drmse = np.sqrt(mean_squared_error(yte_c, pred_c_dummy))

    grad_rrmse = grad_rmse / grad_drmse
    cross_rrmse = cross_rmse / cross_drmse

    grad_corr = np.corrcoef(yte_g, pred_g)[0, 1]
    cross_corr = np.corrcoef(yte_c, pred_c)[0, 1]

    # ---- Save metrics ----
    metrics = pd.DataFrame([{
        "gradient_rmse": grad_rmse,
        "gradient_drmse": grad_drmse,
        "gradient_rrmse": grad_rrmse,
        "gradient_corr": grad_corr,
        "crossfall_rmse": cross_rmse,
        "crossfall_drmse": cross_drmse,
        "crossfall_rrmse": cross_rrmse,
        "crossfall_corr": cross_corr,
    }])

    metrics.to_csv(os.path.join(OUT_METRICS, "metrics_summary.csv"), index=False)
    print(" Saved full metrics → metrics_summary.csv")

    # ---- Save feature importance plot ----
    importance = rf_g.feature_importances_
    feature_names = ["ax", "ay", "az", "gz", "speed"]

    plt.figure(figsize=(6, 4))
    plt.bar(feature_names, importance)
    plt.title("Feature Importance (Gradient Model)")
    plt.ylabel("Importance")
    plt.savefig(os.path.join(OUT_PLOTS, "feature_importance.png"))
    plt.close()

    # ---- Scatter plot: Gradient ----
    plt.figure(figsize=(6, 6))
    plt.scatter(yte_g, pred_g, alpha=0.5)
    plt.xlabel("True Gradient")
    plt.ylabel("Predicted Gradient")
    plt.title("True vs Predicted Gradient")
    plt.savefig(os.path.join(OUT_PLOTS, "gradient_true_vs_pred.png"))
    plt.close()

    # ---- Scatter plot: Crossfall ----
    plt.figure(figsize=(6, 6))
    plt.scatter(yte_c, pred_c, alpha=0.5)
    plt.xlabel("True Crossfall")
    plt.ylabel("Predicted Crossfall")
    plt.title("True vs Predicted Crossfall")
    plt.savefig(os.path.join(OUT_PLOTS, "crossfall_true_vs_pred.png"))
    plt.close()

    print(" Saved plots → plots_rf_v3/")

    # ---- Save models ----
    joblib.dump(rf_g, os.path.join(OUT_MODELS, "rf_gradient.joblib"))
    joblib.dump(rf_c, os.path.join(OUT_MODELS, "rf_crossfall.joblib"))

    return rf_g, rf_c


# ---------------- PER-FRAME PREDICTIONS ----------------
def predict_per_frame(model_g, model_c, df, out_path):
    preds_g, preds_c = [], []

    for i in range(len(df)):
        win = df.iloc[max(0, i - ROLL_WINDOW_FRAMES):i + 1]
        feat = win[["ax", "ay", "az", "gz", "speed"]].mean().to_frame().T

        preds_g.append(model_g.predict(feat)[0])
        preds_c.append(model_c.predict(feat)[0])

    out_df = pd.DataFrame({
        "MT": df["MT"],
        "FrameNumber": np.arange(1, len(df) + 1),
        "predicted_gradient": preds_g,
        "predicted_crossfall": preds_c
    })

    out_df.to_csv(out_path, index=False)
    print(f" Saved prediction → {out_path}")


# ---------------- MAIN ----------------
def main():
    geom_df = load_geometry(GEOM_PATH)
    all_matches = []

    for fname in JSON_FILES:
        print(f"\nProcessing: {fname}")

        df = load_json_fast(os.path.join(DATA_DIR, fname))
        matched = match_accel_to_geometry_fast(df, geom_df, MAX_DISTANCE_METERS)

        print(f"  Matched samples: {len(matched)}")
        all_matches.append(matched)

    full_df = pd.concat(all_matches, ignore_index=True)
    print(f"\nTotal matched samples: {len(full_df)}")

    model_g, model_c = train_rf(full_df)

    print("\nTraining completed successfully!")

    # Per-frame predictions
    for fname in JSON_FILES:
        df = load_json_fast(os.path.join(DATA_DIR, fname))
        out_csv = os.path.join(
            OUT_PERFRAME, f"{os.path.splitext(fname)[0]}_perframe_rf.csv"
        )
        predict_per_frame(model_g, model_c, df, out_csv)


if __name__ == "__main__":
    main()

