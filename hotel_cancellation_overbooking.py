"""
Hotel Cancellation Prediction & Overbooking Revenue Optimization
=================================================================
ไปป์ไลน์ครบวงจร:
  1) Data Preprocessing & Cleaning  (Missing values, Leakage, Outliers)
  2) Feature Engineering            (total_stay, total_guests, is_room_changed, ...)
  3) Encoding                       (One-Hot / Ordinal / Target Encoding)
  4) Modeling                       (Logistic Regression baseline + LightGBM)
                                    + Probability Calibration (Isotonic)
  5) Overbooking Simulation         (Monte Carlo, เทียบ 3 กลยุทธ์)

วิธีรัน:
  pip install pandas numpy scikit-learn lightgbm matplotlib
  python hotel_cancellation_overbooking.py --data hotel_bookings.csv --out outputs

หมายเหตุด้านการออกแบบ:
  * แบ่งข้อมูลตาม "เวลา" (Time-based split) แทนการสุ่ม เพราะในการใช้งานจริง
    เราต้องพยากรณ์การจองในอนาคตจากข้อมูลในอดีต การสุ่มแบ่งจะทำให้ผลดูดีเกินจริง
  * ความน่าจะเป็นที่ได้ต้อง "calibrated" เพราะโมเดล Overbooking ใช้ค่า p
    ไปคำนวณจำนวนแขกที่คาดว่าจะมาจริง ไม่ได้ใช้แค่ลำดับ (ranking)
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, TargetEncoder

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # fallback ถ้าไม่มี LightGBM
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_LGB = False

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

MONTH_MAP = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


# =============================================================================
# 1) DATA PREPROCESSING & CLEANING
# =============================================================================
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[Load] {df.shape[0]:,} rows x {df.shape[1]} cols | cancel rate = {df.is_canceled.mean():.2%}")

    # ---- 1.1 รายงานค่าสูญหาย ----
    na = df.isna().sum()
    print("[Missing] ก่อนจัดการ:\n" + na[na > 0].to_string())

    # ---- 1.2 จัดการค่าสูญหาย ----
    # children: หายแค่ 4 แถว -> ถือว่าไม่มีเด็ก
    df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0).astype(int)
    # country: หมวดหมู่ใหม่ "Unknown" (การไม่ระบุประเทศอาจมีความหมายเชิงพฤติกรรม)
    df["country"] = df["country"].fillna("Unknown")
    # agent: NaN = จองโดยไม่ผ่านเอเจนต์ -> เก็บเป็น flag + รหัส "none"
    df["has_agent"] = df["agent"].notna().astype(int)
    df["agent"] = df["agent"].fillna(0).astype(int).astype(str).replace("0", "none")
    # company: หายถึง ~94% -> ใช้แค่ flag ว่าเป็นการจองแบบบริษัทหรือไม่ แล้วทิ้งรหัส
    df["has_company"] = df["company"].notna().astype(int)
    df = df.drop(columns=["company"])

    # ---- 1.3 ป้องกัน Data Leakage ----
    # reservation_status = ผลลัพธ์สุดท้าย (Canceled / Check-Out / No-Show) = คำตอบโดยตรง
    # reservation_status_date = วันที่สถานะเปลี่ยน (รู้หลังเหตุการณ์เกิดแล้ว)
    df = df.drop(columns=["reservation_status", "reservation_status_date"])

    # ---- 1.4 ข้อมูลผิดปกติ ----
    n0 = len(df)
    df = df[df["adr"] >= 0]                                   # ราคาติดลบ
    df = df[df["adr"] < 1000]                                 # outlier (มีค่า 5,400)
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]  # การจองไม่มีแขก
    print(f"[Clean] ตัดข้อมูลผิดปกติ {n0 - len(df):,} แถว -> เหลือ {len(df):,}")

    # ---- วันที่เข้าพัก (ใช้สำหรับ time-split และ simulation เท่านั้น) ----
    df["arrival_month_num"] = df["arrival_date_month"].map(MONTH_MAP)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df.arrival_date_year, month=df.arrival_month_num, day=df.arrival_date_day_of_month))
    return df.reset_index(drop=True)


# =============================================================================
# 2) FEATURE ENGINEERING
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    # หมายเหตุ: assigned_room_type มักถูกกำหนด ณ วันเช็คอิน -> เป็น "soft leakage"
    # (การจองที่ยกเลิกไปแล้วมักไม่ถูกเปลี่ยนห้อง) ใช้ตาม requirement แต่ควรระวังใน production
    df["is_room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

    # Feature เพิ่มเติมที่มีความหมายเชิงธุรกิจ
    df["has_kids"] = ((df["children"] + df["babies"]) > 0).astype(int)
    df["weekend_ratio"] = np.where(df["total_stay"] > 0,
                                   df["stays_in_weekend_nights"] / df["total_stay"].clip(lower=1), 0)
    df["revenue_est"] = df["adr"] * df["total_stay"]
    df["adr_per_guest"] = df["adr"] / df["total_guests"].clip(lower=1)
    hist = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_ratio"] = np.where(hist > 0, df["previous_cancellations"] / hist.clip(lower=1), 0)
    df["lead_time_log"] = np.log1p(df["lead_time"])
    df["is_deposit_nonrefund"] = (df["deposit_type"] == "Non Refund").astype(int)
    # Cyclical encoding ของเดือน/สัปดาห์ (ธ.ค. ใกล้กับ ม.ค.)
    df["month_sin"] = np.sin(2 * np.pi * df["arrival_month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["arrival_month_num"] / 12)
    df["arrival_dow"] = df["arrival_date"].dt.dayofweek
    return df


# =============================================================================
# 3) ENCODING  (เลือกวิธีตามลักษณะตัวแปร)
# =============================================================================
TARGET = "is_canceled"

# One-Hot: หมวดหมู่น้อย ไม่มีลำดับ
OHE_COLS = ["hotel", "meal", "market_segment", "distribution_channel",
            "deposit_type", "customer_type"]
# Ordinal: ประเภทห้อง A..L มีลำดับ (โดยประมาณสะท้อนระดับห้อง)
ORD_COLS = ["reserved_room_type", "assigned_room_type"]
# Target Encoding: หมวดหมู่มาก (country ~177, agent ~330) -> OHE จะบวมเกินไป
# sklearn TargetEncoder ทำ cross-fitting ภายในตอน fit_transform -> ลด target leakage
TE_COLS = ["country", "agent"]
NUM_COLS = [
    "lead_time", "lead_time_log", "arrival_month_num", "arrival_date_week_number",
    "arrival_date_day_of_month", "arrival_dow", "month_sin", "month_cos",
    "stays_in_weekend_nights", "stays_in_week_nights", "total_stay", "weekend_ratio",
    "adults", "children", "babies", "total_guests", "has_kids",
    "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled",
    "prev_cancel_ratio", "booking_changes", "days_in_waiting_list",
    "adr", "adr_per_guest", "revenue_est", "required_car_parking_spaces",
    "total_of_special_requests", "is_room_changed", "has_agent", "has_company",
    "is_deposit_nonrefund",
]
# ไม่ใช้ arrival_date_year เป็นฟีเจอร์: เมื่อแบ่งตามเวลา ปีใน test จะไม่เคยเห็นในช่วงต้นของ train
FEATURES = OHE_COLS + ORD_COLS + TE_COLS + NUM_COLS


def build_preprocessor(scale: bool) -> ColumnTransformer:
    num_step = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False), OHE_COLS),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ORD_COLS),
        ("te", TargetEncoder(target_type="binary", cv=5), TE_COLS),
        ("num", num_step, NUM_COLS),
    ], verbose_feature_names_out=False)


# =============================================================================
# 4) MODELING
# =============================================================================
def time_split(df: pd.DataFrame):
    """Train: ก.ค.2015–ธ.ค.2016 | Calibration: ม.ค.–เม.ย.2017 | Test: พ.ค.–ส.ค.2017"""
    train = df[df.arrival_date < "2017-01-01"]
    calib = df[(df.arrival_date >= "2017-01-01") & (df.arrival_date < "2017-05-01")]
    test = df[df.arrival_date >= "2017-05-01"]
    for n, d in [("train", train), ("calib", calib), ("test", test)]:
        print(f"[Split] {n:5s}: {len(d):>6,} rows | {d.arrival_date.min().date()} → "
              f"{d.arrival_date.max().date()} | cancel={d[TARGET].mean():.2%}")
    return train, calib, test


def build_models() -> dict[str, Pipeline]:
    models = {
        "LogisticRegression": Pipeline([
            ("prep", build_preprocessor(scale=True)),
            ("clf", LogisticRegression(max_iter=2000, C=0.5)),
        ])
    }
    if HAS_LGB:
        gbm = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.03, num_leaves=63, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
            reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1)
    else:
        gbm = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05,
                                             max_leaf_nodes=63, random_state=RANDOM_STATE)
    models["LightGBM" if HAS_LGB else "HistGB"] = Pipeline([
        ("prep", build_preprocessor(scale=False)), ("clf", gbm)])
    return models


def evaluate(name: str, y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    return {
        "model": name,
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Brier": brier_score_loss(y, p),
        "F1@thr": f1_score(y, p >= thr),
        "pred_cancel_rate": p.mean(),
        "actual_cancel_rate": y.mean(),
    }


def train_and_select(train, calib, test, out_dir):
    X_tr, y_tr = train[FEATURES], train[TARGET].values
    X_ca, y_ca = calib[FEATURES], calib[TARGET].values
    X_te, y_te = test[FEATURES], test[TARGET].values

    results, fitted, calibrators = [], {}, {}
    for name, pipe in build_models().items():
        pipe.fit(X_tr, y_tr)
        p_ca_raw = pipe.predict_proba(X_ca)[:, 1]
        # Isotonic calibration บนช่วงเวลาที่ไม่ได้ใช้เทรน -> แก้ drift ของอัตรายกเลิก
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_ca_raw, y_ca)
        p_te_raw = pipe.predict_proba(X_te)[:, 1]
        p_te = iso.predict(p_te_raw)
        results.append(evaluate(name + " (raw)", y_te, p_te_raw))
        results.append(evaluate(name + " (calibrated)", y_te, p_te))
        fitted[name], calibrators[name] = pipe, iso

    res = pd.DataFrame(results).set_index("model")
    print("\n[Model] ผลบน Test set (พ.ค.–ส.ค. 2017):\n" + res.round(4).to_string())
    res.to_csv(os.path.join(out_dir, "model_metrics.csv"))

    best = max(fitted, key=lambda n: res.loc[n + " (calibrated)", "ROC_AUC"])
    pipe, iso = fitted[best], calibrators[best]
    p_te = iso.predict(pipe.predict_proba(X_te)[:, 1])
    print(f"\n[Model] เลือกโมเดล: {best}")
    print(classification_report(y_te, p_te >= 0.5, target_names=["Not canceled", "Canceled"], digits=3))

    save_feature_importance(pipe, best, out_dir)
    save_model_plots(y_te, p_te, best, out_dir)
    return best, pipe, iso, p_te


def save_feature_importance(pipe, name, out_dir, top=20):
    clf = pipe.named_steps["clf"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(clf, "booster_"):
        imp = clf.booster_.feature_importance(importance_type="gain")
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        return
    fi = pd.Series(imp, index=names).sort_values(ascending=False)
    fi = fi / fi.sum()
    fi.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["importance"])
    print("[Model] Top 10 features:\n" + fi.head(10).round(4).to_string())
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        fi.head(top)[::-1].plot.barh(ax=ax, color="#3b6ea5")
        ax.set_title(f"Top {top} Feature Importance (gain share) – {name}")
        ax.set_xlabel("Share of total gain"); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=130); plt.close(fig)
    except ImportError:
        pass


def save_model_plots(y, p, name, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import roc_curve
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr, color="#3b6ea5", lw=2, label=f"AUC = {roc_auc_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set(title=f"ROC Curve – {name}", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend(loc="lower right")
    frac, mean_p = calibration_curve(y, p, n_bins=15, strategy="quantile")
    axes[1].plot(mean_p, frac, "o-", color="#3b6ea5", label="Model")
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    axes[1].set(title="Calibration (Reliability) Curve", xlabel="Predicted P(cancel)",
                ylabel="Observed cancel rate")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "model_roc_calibration.png"), dpi=130)
    plt.close(fig)


# =============================================================================
# 5) OVERBOOKING SIMULATION & REVENUE OPTIMIZATION
# =============================================================================
@dataclass
class OverbookingConfig:
    capacity_quantile: float = 0.90   # ประมาณจำนวนห้องจริง = P90 ของห้องที่มีแขกเข้าพักจริงต่อคืน
    walk_cost_multiplier: float = 2.0 # ต้นทุน "walk" แขก 1 ห้อง = 2 x ADR เฉลี่ย (ห้องโรงแรมอื่น+ค่าเดินทาง+ชดเชย)
    walk_fixed_cost: float = 50.0     # ค่าเสียชื่อเสียง/goodwill ต่อการ walk 1 ครั้ง
    max_overbook_pct: float = 1.00    # เพดาน booking limit = C x (1 + 100%) (กันโมเดลพลาดหนัก)
    n_sims: int = 3000                # จำนวนรอบ Monte Carlo ต่อคืน


def expand_to_nights(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """แตกการจอง 1 รายการ -> 1 แถวต่อคืนที่พัก (room-night) เพื่อคิด occupancy รายคืน"""
    d = df[["hotel", "arrival_date", "lead_time", "total_stay", "adr", TARGET]].copy()
    d["p_cancel"] = p
    d["booking_date"] = d["arrival_date"] - pd.to_timedelta(d["lead_time"], unit="D")
    d["n_nights"] = d["total_stay"].clip(lower=1)  # day-use (0 คืน) นับเป็น 1 คืน
    rep = d.loc[d.index.repeat(d["n_nights"])].copy()
    rep["night"] = rep["arrival_date"] + pd.to_timedelta(rep.groupby(level=0).cumcount(), unit="D")
    return rep.reset_index(drop=True)


def estimate_capacity(nights_all: pd.DataFrame, q: float) -> dict:
    occ = (nights_all[nights_all[TARGET] == 0].groupby(["hotel", "night"]).size())
    return {h: int(np.ceil(occ.loc[h].quantile(q))) for h in occ.index.get_level_values(0).unique()}


def _profit_curves(adr, show_matrix, C, walk_unit_cost):
    """profit ของทุก booking limit L (L=1..N) พร้อมกัน ด้วย cumulative sum.
    show_matrix: (n_sims, N) โดยเรียงตามลำดับการจอง (มาก่อนได้ก่อน)"""
    shows_cum = np.cumsum(show_matrix, axis=1)                    # จำนวนแขกที่มาจริง ถ้ารับ L รายการ
    rev_cum = np.cumsum(show_matrix * adr[None, :], axis=1)       # รายได้ถ้ารับแขกทุกคนที่มา
    overflow = np.maximum(shows_cum - C, 0)                       # แขกที่ต้อง walk
    # แขกที่ถูก walk ไม่ได้จ่ายค่าห้อง -> หัก ADR เฉลี่ยของแขกที่มา + ค่า walk
    avg_adr = np.divide(rev_cum, np.maximum(shows_cum, 1))
    profit = rev_cum - overflow * (avg_adr + walk_unit_cost)
    return profit, overflow, np.minimum(shows_cum, C)


def simulate_overbooking(test_df, p_test, hist_cancel_rate, capacity, cfg: OverbookingConfig, out_dir):
    rng = np.random.default_rng(RANDOM_STATE)
    nights = expand_to_nights(test_df, p_test)
    # ประเมินเฉพาะคืนที่อยู่ในช่วง test ทั้งหมด (ตัดคืนท้ายที่อาจตกหล่นการจองถัดไป)
    nights = nights[nights.night <= test_df.arrival_date.max()]

    rows = []
    for (hotel, night), g in nights.groupby(["hotel", "night"]):
        g = g.sort_values("booking_date")              # first-come, first-served
        adr = g["adr"].values
        actual_show = (1 - g[TARGET].values).astype(float)
        p = g["p_cancel"].values
        N, C = len(g), capacity[hotel]
        walk_cost = cfg.walk_cost_multiplier * adr.mean() + cfg.walk_fixed_cost

        # ---- กลยุทธ์ที่ 1: ไม่ overbook ----
        L_none = min(C, N)
        # ---- กลยุทธ์ที่ 2: overbook แบบคงที่ จากอัตรายกเลิกเฉลี่ยในอดีต ----
        L_flat = min(N, int(np.floor(C / (1 - hist_cancel_rate[hotel]))))
        # ---- กลยุทธ์ที่ 3: ML-driven -> เลือก L ที่ให้ Expected Profit สูงสุด ----
        sims = (rng.random((cfg.n_sims, N)) > p[None, :]).astype(np.float32)  # 1 = แขกมา
        exp_profit, _, _ = _profit_curves(adr, sims, C, walk_cost)
        exp_profit = exp_profit.mean(axis=0)
        L_max = min(N, int(C * (1 + cfg.max_overbook_pct)))
        L_ml = int(np.argmax(exp_profit[:L_max])) + 1 if N > 0 else 0
        L_ml = max(L_ml, L_none)  # ไม่ต่ำกว่าการไม่ overbook (ห้องว่างไม่ทำเงิน)

        # ---- วัดผลด้วย "ผลลัพธ์จริง" (is_canceled จริง) ----
        act_profit, act_over, act_occ = _profit_curves(adr, actual_show[None, :], C, walk_cost)
        for strat, L in [("1_No_Overbooking", L_none), ("2_Flat_Historical", L_flat), ("3_ML_Optimized", L_ml)]:
            i = L - 1
            rows.append(dict(hotel=hotel, night=night, strategy=strat, capacity=C, demand=N,
                             booking_limit=L, profit=act_profit[0, i], walked=act_over[0, i],
                             occupied=act_occ[0, i], ml_expected_profit=exp_profit[i]))

    sim = pd.DataFrame(rows)
    sim.to_csv(os.path.join(out_dir, "overbooking_nightly.csv"), index=False)

    summary = (sim.groupby(["hotel", "strategy"])
               .agg(nights=("night", "nunique"), net_revenue=("profit", "sum"),
                    avg_booking_limit=("booking_limit", "mean"), total_walked=("walked", "sum"),
                    nights_with_walk=("walked", lambda s: int((s > 0).sum())),
                    occupancy=("occupied", "sum"), cap=("capacity", "sum")))
    summary["occupancy_rate"] = summary.pop("occupancy") / summary.pop("cap")
    base = summary.xs("1_No_Overbooking", level="strategy")["net_revenue"]
    summary["uplift_vs_no_ob"] = summary["net_revenue"] / summary.index.get_level_values(0).map(base) - 1
    print("\n[Overbooking] สรุปผลบนช่วง Test (วัดด้วยการยกเลิกที่เกิดขึ้นจริง):")
    with pd.option_context("display.float_format", "{:,.3f}".format):
        print(summary.to_string())
    summary.to_csv(os.path.join(out_dir, "overbooking_summary.csv"))
    plot_overbooking(sim, out_dir)
    return summary, sim


def plot_overbooking(sim, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"1_No_Overbooking": "#9aa5b1", "2_Flat_Historical": "#e3a33b", "3_ML_Optimized": "#3b6ea5"}
    hotels = sim.hotel.unique()
    fig, axes = plt.subplots(len(hotels), 1, figsize=(12, 4.2 * len(hotels)), sharex=True)
    for ax, h in zip(np.atleast_1d(axes), hotels):
        s = sim[sim.hotel == h]
        cap = s.capacity.iloc[0]
        dem = s.drop_duplicates("night").set_index("night")["demand"]
        ax.fill_between(dem.index, dem.values, color="#dfe6ee", label="Total bookings (demand)")
        for k, c in colors.items():
            t = s[s.strategy == k].set_index("night")["booking_limit"]
            ax.plot(t.index, t.values, color=c, lw=1.6, label=k.split("_", 1)[1].replace("_", " "))
        ax.axhline(cap, color="black", ls="--", lw=1, label=f"Capacity ≈ {cap}")
        ax.set_title(f"{h}: Nightly booking limit by strategy"); ax.set_ylabel("Rooms")
        ax.legend(loc="upper left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "overbooking_limits.png"), dpi=130); plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="hotel_bookings.csv")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--walk-mult", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 70 + "\n STEP 1-2: Cleaning & Feature Engineering\n" + "=" * 70)
    df = engineer_features(load_and_clean(args.data))
    assert not {"reservation_status", "reservation_status_date"} & set(df.columns), "Leakage!"
    assert df[FEATURES].isna().sum().sum() == 0, "ยังมีค่าว่างในฟีเจอร์"

    print("\n" + "=" * 70 + "\n STEP 3-4: Encoding & Modeling\n" + "=" * 70)
    train, calib, test = time_split(df)
    best, pipe, iso, p_test = train_and_select(train, calib, test, args.out)

    pred = test[["hotel", "arrival_date", "lead_time", "adr", "deposit_type", TARGET]].copy()
    pred["p_cancel"] = p_test
    pred["risk_band"] = pd.cut(p_test, [0, .2, .5, .8, 1], labels=["Low", "Medium", "High", "Very high"],
                               include_lowest=True)
    pred.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)
    print("[Model] อัตรายกเลิกจริงตาม Risk band:\n" +
          pred.groupby("risk_band", observed=True)[TARGET].agg(["count", "mean"]).round(3).to_string())

    print("\n" + "=" * 70 + "\n STEP 5: Overbooking Simulation\n" + "=" * 70)
    cfg = OverbookingConfig(walk_cost_multiplier=args.walk_mult)
    history = pd.concat([train, calib])
    capacity = estimate_capacity(expand_to_nights(df, np.zeros(len(df))), cfg.capacity_quantile)
    hist_rate = history.groupby("hotel")[TARGET].mean().to_dict()
    print(f"[Overbooking] ประมาณจำนวนห้อง (P{int(cfg.capacity_quantile*100)} ของห้องที่มีแขกจริง/คืน): {capacity}")
    print(f"[Overbooking] อัตรายกเลิกในอดีต: { {k: round(v, 3) for k, v in hist_rate.items()} }")
    simulate_overbooking(test, p_test, hist_rate, capacity, cfg, args.out)
    print(f"\nเสร็จสิ้น — ไฟล์ผลลัพธ์อยู่ในโฟลเดอร์: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
