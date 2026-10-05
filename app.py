"""
แอปทำนายการยกเลิกการจองห้องพัก (ธีมสนุก สีสันสดใส)
วิธีรัน:   python -m streamlit run app.py
ต้องมี hotel_cancellation_overbooking.py และ hotel_bookings.csv อยู่โฟลเดอร์เดียวกัน
"""
from __future__ import annotations

import datetime as dt
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import hotel_cancellation_overbooking as core

st.set_page_config(page_title="ทายใจนักท่องเที่ยว", page_icon="🏨", layout="wide",
                   initial_sidebar_state="collapsed")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(APP_DIR, "outputs")
BUNDLE_PATH = os.path.join(OUT_DIR, "predict_model_real.joblib")
DATA_PATH = os.path.join(APP_DIR, "hotel_bookings.csv")

# ข้อมูลต้นฉบับเก็บราคาเป็นยูโร -> ผู้ใช้กรอกเป็นบาท แล้วระบบแปลงเป็นยูโรก่อนส่งเข้าโมเดล
# อัตราแลกเปลี่ยนกลาง (XE) ณ 5 ต.ค. 2026: 1 EUR ≈ 37.77 THB  (แก้ตัวเลขนี้ได้ตามอัตราปัจจุบัน)
EUR_TO_THB = 37.77

# ปรับผลตาม "หลักความเป็นจริงทางธุรกิจ" เรื่องเงินมัดจำ
# ในข้อมูลชุดนี้ "มัดจำแบบไม่คืนเงิน" ยกเลิกถึง 99% เพราะส่วนใหญ่เป็นการเหมาห้องของบริษัททัวร์
# ซึ่งไม่ตรงกับพฤติกรรมลูกค้าทั่วไป จึงให้โมเดลทำนายแบบ "ไม่มีมัดจำ" ก่อน แล้วปรับด้วยตัวคูณ odds:
#   ค่า < 1 = ลดโอกาสยกเลิก, 1 = ไม่ปรับ  (แก้ตัวเลขได้ตามที่ต้องการ)
DEPOSIT_ODDS = {"No Deposit": 1.0,     # ไม่มีมัดจำ: ใช้ผลโมเดลตามจริง
                "Refundable": 0.4,     # มัดจำขอคืนได้: มีภาระผูกพันบ้าง -> ยกเลิกน้อยลง
                "Non Refund": 0.05}    # มัดจำไม่คืนเงิน: ยกเลิกแล้วเสียเงิน -> เข้าพักเกือบแน่นอน


def adjust_for_deposit(p: float, deposit: str) -> float:
    odds = p / max(1 - p, 1e-9) * DEPOSIT_ODDS.get(deposit, 1.0)
    return odds / (1 + odds)

# ---------------------------------------------------------------- ชื่อภาษาไทย
HOTEL_TH = {"City Hotel": "🏙️  โรงแรม", "Resort Hotel": "🏝️  รีสอร์ท"}
DEPOSIT_TH = {"No Deposit": "🙅  ไม่มีเงินมัดจำ", "Non Refund": "🔒  มัดจำแบบไม่คืนเงิน",
              "Refundable": "💸  มัดจำแบบขอคืนเงินได้"}
SEGMENT_TH = {"Online TA": "📱  ตัวแทนท่องเที่ยวออนไลน์", "Offline TA/TO": "🧳  บริษัททัวร์ / ตัวแทนท่องเที่ยว",
              "Direct": "☎️  จองตรงกับโรงแรม", "Groups": "👨‍👩‍👧‍👦  การจองแบบหมู่คณะ",
              "Corporate": "🏢  องค์กร / บริษัท", "Aviation": "✈️  สายการบิน"}
COUNTRY_TH = {"PRT": "โปรตุเกส", "GBR": "สหราชอาณาจักร", "FRA": "ฝรั่งเศส", "ESP": "สเปน",
              "DEU": "เยอรมนี", "ITA": "อิตาลี", "IRL": "ไอร์แลนด์", "BEL": "เบลเยียม",
              "BRA": "บราซิล", "NLD": "เนเธอร์แลนด์", "THA": "ไทย"}
COUNTRY_FLAG = {"PRT": "🇵🇹", "GBR": "🇬🇧", "FRA": "🇫🇷", "ESP": "🇪🇸", "DEU": "🇩🇪", "ITA": "🇮🇹",
                "IRL": "🇮🇪", "BEL": "🇧🇪", "BRA": "🇧🇷", "NLD": "🇳🇱", "THA": "🇹🇭"}
TH_MONTHS = ["ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.", "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค."]


# ---------------------------------------------------------------- ตกแต่งหน้าเว็บ
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Mali:wght@500;600;700&family=Mitr:wght@300;400;500&family=Noto+Color+Emoji&display=swap');
:root { --pink:#ff6fae; --orange:#ff9f43; --yellow:#ffd93d; --mint:#3ddc97; --sky:#4cc9f0;
        --purple:#8b5cf6; --ink:#3b2a5c; --muted:#7c6f99; }
html, body, .stApp, button, input, label, p, span, div { font-family:'Mitr','Noto Color Emoji',sans-serif !important; }
h1, h2, h3, .fun { font-family:'Mali','Noto Color Emoji',cursive !important; }
#MainMenu, header[data-testid="stHeader"], footer, .stDeployButton, [data-testid="stToolbar"],
[data-testid="InputInstructions"] { display:none !important; }

.stApp { background: linear-gradient(160deg, #fff4fb 0%, #f1f0ff 45%, #e6f8ff 100%); }
.block-container { padding-top:1.2rem; padding-bottom:2rem; max-width:1180px; }

/* ลอยไปมา */
.floaty { position:fixed; font-size:2.4rem; opacity:.5; z-index:0; pointer-events:none;
          animation:float 7s ease-in-out infinite; }
@keyframes float { 0%,100% { transform:translateY(0) rotate(-4deg); } 50% { transform:translateY(-22px) rotate(6deg); } }
@keyframes bounce { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-10px); } }
@keyframes wiggle { 0%,100% { transform:rotate(0); } 25% { transform:rotate(-8deg); } 75% { transform:rotate(8deg); } }
@keyframes pop { 0% { opacity:0; transform:scale(.85); } 70% { transform:scale(1.03); } 100% { opacity:1; transform:scale(1); } }

/* ---------- hero ---------- */
.hero { position:relative; z-index:1; border-radius:32px; padding:30px 36px; margin-bottom:24px; color:#fff;
        background:linear-gradient(120deg, var(--purple) 0%, var(--pink) 55%, var(--orange) 100%);
        box-shadow:0 14px 0 rgba(139,92,246,.18), 0 24px 40px rgba(255,111,174,.30);
        display:flex; align-items:center; gap:26px; flex-wrap:wrap; overflow:hidden; }
.hero::after { content:"⭐  ✨  ⭐"; position:absolute; right:30px; top:16px; font-size:1.4rem; opacity:.75; }
.hero .mascot { font-size:4.6rem; animation:bounce 2.2s ease-in-out infinite;
                filter:drop-shadow(0 8px 10px rgba(0,0,0,.18)); }
.hero h1 { color:#fff; font-size:2.35rem; margin:0; padding:0; line-height:1.25;
           text-shadow:0 3px 0 rgba(0,0,0,.12); }
.hero p { margin:6px 0 0; font-size:1.05rem; color:#fff6fb; }
.pill { display:inline-block; background:#fff; color:var(--pink); font-weight:500; border-radius:99px;
        padding:3px 14px; font-size:.82rem; margin-bottom:10px; box-shadow:0 4px 0 rgba(0,0,0,.08); }

/* ---------- form ---------- */
div[data-testid="stForm"] { position:relative; z-index:1; background:#fff; border:3px solid #efe7ff;
        border-radius:30px; padding:26px 28px 24px; box-shadow:0 10px 0 #ece4ff, 0 22px 40px rgba(139,92,246,.12); }
.sec { display:flex; align-items:center; gap:12px; margin:2px 0 8px; }
.sec .num { width:46px; height:46px; border-radius:16px; display:flex; align-items:center; justify-content:center;
            font-size:1.5rem; animation:wiggle 3s ease-in-out infinite; }
.sec .t { font-family:'Mali',cursive; font-weight:700; color:var(--ink); font-size:1.3rem; line-height:1.15; }
.sec .d { font-size:.82rem; color:var(--muted); font-weight:300; }
.s1 .num { background:#ffe3f0; } .s2 .num { background:#dff6ff; }
.gap { height:10px; border-bottom:3px dotted #f0e9ff; margin:2px 0 18px; }
label p { font-size:.95rem !important; color:var(--ink) !important; font-weight:400 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] {
        border-radius:16px !important; background:#faf8ff !important; border:2px solid #ebe3ff !important;
        transition:border-color .15s, box-shadow .15s; }
div[data-baseweb="select"] > div:hover, div[data-baseweb="input"]:hover,
div[data-baseweb="input"]:focus-within { border-color:var(--purple) !important;
        box-shadow:0 0 0 4px rgba(139,92,246,.12); }
div[data-baseweb="input"] input { background:transparent !important; }
button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"] {
        border-radius:12px !important; color:var(--purple) !important; }

div[data-testid="stFormSubmitButton"] button { height:3.8rem; border-radius:22px; border:none;
        background:linear-gradient(90deg, var(--mint), var(--sky)); box-shadow:0 8px 0 #22a6a0;
        transition:transform .12s, box-shadow .12s; }
div[data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); box-shadow:0 10px 0 #22a6a0; }
div[data-testid="stFormSubmitButton"] button:active { transform:translateY(6px); box-shadow:0 2px 0 #22a6a0; }
div[data-testid="stFormSubmitButton"] button p { color:#fff !important; font-family:'Mali',cursive !important;
        font-size:1.4rem !important; font-weight:700 !important; text-shadow:0 2px 0 rgba(0,0,0,.12); }

/* ---------- result ---------- */
.card { position:relative; z-index:1; background:#fff; border:3px solid #efe7ff; border-radius:30px;
        padding:26px; text-align:center; box-shadow:0 10px 0 #ece4ff, 0 22px 40px rgba(139,92,246,.12);
        animation:pop .5s ease; }
.wait .big { font-size:5rem; animation:bounce 2s ease-in-out infinite; }
.wait h3 { color:var(--ink); margin:6px 0 4px; font-size:1.6rem; }
.wait p { color:var(--muted); margin:0 0 18px; }
.steps { text-align:left; display:grid; gap:10px; }
.step { display:flex; gap:12px; align-items:center; border-radius:18px; padding:12px 14px; color:var(--ink); }
.step i { font-style:normal; font-size:1.5rem; }
.st1 { background:#fff0f7; } .st2 { background:#eef9ff; } .st3 { background:#effcf5; }

.face { font-size:5.2rem; line-height:1; animation:bounce 1.8s ease-in-out infinite; }
.verdict { font-family:'Mali',cursive; font-size:2.3rem; font-weight:700; margin:8px 0 0; }
.vsub { color:var(--muted); margin-bottom:16px; }
.meter { position:relative; height:30px; border-radius:99px; overflow:hidden; margin:6px 0 6px;
         background:#f2eefc; border:3px solid #ebe3ff; }
.meter > div { height:100%; border-radius:99px; display:flex; align-items:center; justify-content:flex-end;
               padding-right:12px; color:#fff; font-weight:500; font-size:.95rem; min-width:56px;
               background-size:28px 28px !important; animation:grow 1s ease; }
@keyframes grow { from { width:0; } }
.scale { display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-bottom:16px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-bottom:14px; }
.chip { background:#f7f4ff; border-radius:99px; padding:6px 12px; font-size:.86rem; color:var(--ink); }
.money { border-radius:22px; padding:14px 18px; background:linear-gradient(120deg, #fff6d6, #ffe9f3);
         display:flex; justify-content:space-between; align-items:center; }
.money span { color:var(--muted); font-size:.88rem; text-align:left; }
.money b { font-family:'Mali',cursive; color:#e8590c; font-size:1.7rem; }
.why { text-align:left; background:#fbf9ff; border-radius:20px; padding:12px 14px; margin-bottom:14px; }
.why-t { font-weight:500; color:var(--ink); margin-bottom:6px; }
.r { display:flex; gap:10px; align-items:flex-start; padding:5px 0; font-size:.9rem; color:var(--ink); }
.r i { font-style:normal; flex:none; width:24px; height:24px; border-radius:8px; display:flex;
       align-items:center; justify-content:center; font-size:.8rem; font-weight:700; color:#fff; }
.r i.up { background:#ff4d6d; } .r i.down { background:#12b76a; }
.foot { text-align:center; color:var(--muted); font-size:.82rem; margin-top:26px; }
</style>
<div class="floaty" style="left:2%;top:18%">☁️</div>
<div class="floaty" style="right:3%;top:42%;animation-delay:1.5s">🎈</div>
<div class="floaty" style="left:3%;bottom:10%;animation-delay:3s">🌴</div>
<div class="floaty" style="right:6%;bottom:6%;animation-delay:2s">✈️</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- โมเดล
# โมเดลนี้เรียนรู้จาก "ข้อมูลที่ผู้ใช้กรอกได้จริงเท่านั้น" (ไม่มีค่าซ่อนที่เดาให้)
# และเรียนจากการจองแบบ "ไม่มีมัดจำ" เท่านั้น แล้วค่อยปรับตามประเภทมัดจำด้วยกฎธุรกิจ (DEPOSIT_ODDS)
# เพื่อตัดความเอนเอียงของข้อมูล (มัดจำไม่คืนเงินในข้อมูลชุดนี้ = บริษัททัวร์เหมาห้องแล้วยกเลิก 99%)
FEATS = ["hotel", "market_segment", "country", "arrival_month", "lead_time", "nights", "weekend_nights", "adr"]
CATS = ["hotel", "market_segment", "country"]
MONO = [0, 0, 0, 0, 1, 0, 0, 0]   # จองล่วงหน้านานขึ้น -> โอกาสยกเลิกต้องไม่ลดลง (ตามสามัญสำนึก)


def make_X(df: pd.DataFrame, cats: dict) -> pd.DataFrame:
    X = pd.DataFrame({
        "hotel": df["hotel"], "market_segment": df["market_segment"],
        "country": df["country"].where(df["country"].isin(COUNTRY_TH), "OTHER"),
        "arrival_month": df["arrival_month"].astype(int), "lead_time": df["lead_time"].astype(float),
        "nights": df["nights"].astype(float), "weekend_nights": df["weekend_nights"].astype(float),
        "adr": df["adr"].astype(float)})
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=cats[c])
    return X[FEATS]


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrate(p, a: float, b: float):
    """Platt scaling: ปรับความน่าจะเป็นให้ตรงกับอัตรายกเลิกจริง (เส้นโค้งเรียบ)"""
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


@st.cache_resource(show_spinner="🧠 กำลังสอนให้ระบบฉลาด (ครั้งแรกประมาณ 30 วินาที)...")
def load_model() -> dict:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    os.makedirs(OUT_DIR, exist_ok=True)
    s = os.stat(DATA_PATH)
    sig = (s.st_size, int(s.st_mtime))
    if os.path.exists(BUNDLE_PATH):
        try:
            m = joblib.load(BUNDLE_PATH)
            if m.get("signature") == sig:
                return m
        except Exception:
            pass
    df = core.load_and_clean(DATA_PATH)
    df = df[(df["deposit_type"] == "No Deposit") & (df["adr"] > 0)
            & df["market_segment"].isin(SEGMENT_TH)].copy()
    df["nights"] = (df["stays_in_weekend_nights"] + df["stays_in_week_nights"]).clip(lower=1)
    df["weekend_nights"] = df["stays_in_weekend_nights"]
    df["arrival_month"] = df["arrival_month_num"]
    cats = {"hotel": list(HOTEL_TH), "market_segment": list(SEGMENT_TH), "country": list(COUNTRY_TH) + ["OTHER"]}

    train = df[df.arrival_date < "2017-01-01"]
    calib = df[(df.arrival_date >= "2017-01-01") & (df.arrival_date < "2017-05-01")]
    test = df[df.arrival_date >= "2017-05-01"]
    # max_depth=1 = โมเดลแบบบวกกัน (additive): แต่ละปัจจัยมีผลแยกกันชัดเจน อธิบายได้ตรงไปตรงมา
    # และให้ผลที่สมเหตุสมผลกว่าโมเดลซับซ้อน (ความแม่นยำใกล้เคียงกัน)
    clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, max_depth=1, num_leaves=2,
                             min_child_samples=200, monotone_constraints=MONO, random_state=42, verbose=-1)
    clf.fit(make_X(train, cats), train[core.TARGET])
    from sklearn.linear_model import LogisticRegression
    z = _logit(clf.predict_proba(make_X(calib, cats))[:, 1]).reshape(-1, 1)
    lr = LogisticRegression(C=1e6).fit(z, calib[core.TARGET])
    cal = (float(lr.coef_[0][0]), float(lr.intercept_[0]))
    p_test = calibrate(clf.predict_proba(make_X(test, cats))[:, 1], *cal)
    y = test[core.TARGET].values
    m = dict(signature=sig, clf=clf, cal=cal, cats=cats,
             accuracy=float(((p_test >= 0.5) == y).mean()), auc=float(roc_auc_score(y, p_test)))
    joblib.dump(m, BUNDLE_PATH)
    return m


def predict(m: dict, row: dict) -> tuple[float, dict]:
    """คืนค่า (โอกาสยกเลิก, ผลของแต่ละปัจจัย) — ผลของปัจจัยใช้อธิบายเหตุผลให้คนอ่านเข้าใจ"""
    X = make_X(pd.DataFrame([row]), m["cats"])
    p = float(calibrate(m["clf"].predict_proba(X)[:, 1], *m["cal"])[0])
    contrib = m["clf"].booster_.predict(X, pred_contrib=True)[0][:-1]
    return p, dict(zip(FEATS, contrib))


def explain(contrib, p_base, p, deposit, lead_time, nights, price, hotel, segment, country, arrival) -> str:
    """แปลงผลของแต่ละปัจจัยเป็นประโยคภาษาคน (แสดง 3 ปัจจัยหลัก + เรื่องมัดจำ)"""
    seg, place = SEGMENT_TH[segment].split("  ")[-1], HOTEL_TH[hotel].split("  ")[-1]
    up = {"lead_time": f"จองล่วงหน้า {lead_time} วัน มีเวลาเปลี่ยนใจได้นาน",
          "market_segment": f"จองผ่าน{seg} มักยกเลิกบ่อย",
          "country": f"ลูกค้าจาก{COUNTRY_TH[country]}ยกเลิกค่อนข้างบ่อย",
          "hotel": f"{place}มักถูกยกเลิกบ่อยกว่า",
          "arrival_month": f"เข้าพักเดือน{TH_MONTHS[arrival.month - 1]} คนยกเลิกบ่อย",
          "nights": f"พัก {nights} คืน มักยกเลิกบ่อยกว่า",
          "weekend_nights": "จำนวนคืนวันหยุดแบบนี้ยกเลิกบ่อย",
          "adr": f"ราคา ฿{price:,}/คืน ระดับนี้คนยกเลิกบ่อย"}
    down = {"lead_time": f"จองใกล้วันเข้าพัก ({lead_time} วัน) มักมาจริง",
            "market_segment": f"จองผ่าน{seg} มักมาพักจริง",
            "country": f"ลูกค้าจาก{COUNTRY_TH[country]}มักมาพักจริง",
            "hotel": f"{place}มักถูกยกเลิกน้อยกว่า",
            "arrival_month": f"เข้าพักเดือน{TH_MONTHS[arrival.month - 1]} คนมักมาจริง",
            "nights": f"พัก {nights} คืน มักมาจริง",
            "weekend_nights": "มีคืนวันหยุด ลูกค้ามักมาเที่ยวจริง",
            "adr": f"ราคา ฿{price:,}/คืน ระดับนี้คนมักมาจริง"}
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
    rows = [(v > 0, (up if v > 0 else down)[k]) for k, v in top if abs(v) > 0.05]
    if deposit == "No Deposit":
        rows.append((True, "ไม่มีเงินมัดจำ ยกเลิกได้ฟรี ไม่เสียอะไร"))
    elif deposit == "Refundable":
        rows.append((False, f"มีเงินมัดจำ ลดโอกาสยกเลิกจาก {p_base:.0%} เหลือ {p:.0%}"))
    else:
        rows.append((False, f"มัดจำไม่คืนเงิน ยกเลิกแล้วเสียเงิน ลดจาก {p_base:.0%} เหลือ {p:.0%}"))
    return "".join(f'<div class="r"><i class="{"up" if u else "down"}">{"▲" if u else "▼"}</i>{t}</div>'
                   for u, t in rows)


def thai_date(d: dt.date) -> str:
    return f"{d.day} {TH_MONTHS[d.month - 1]} {d.year + 543}"


def section(cls: str, icon: str, title: str, desc: str) -> None:
    st.markdown(f'<div class="sec {cls}"><div class="num">{icon}</div><div><div class="t">{title}</div>'
                f'<div class="d">{desc}</div></div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- หน้าเว็บ
if not os.path.exists(DATA_PATH):
    st.error("ไม่พบไฟล์ hotel_bookings.csv ในโฟลเดอร์เดียวกับ app.py")
    st.stop()
model = load_model()

st.markdown(f"""<div class="hero"><div class="mascot">🏨</div>
    <h1>การพยากรณ์การยกเลิกการจองห้องพัก</h1>
    <p>ลูกค้าคนนี้จะมาพักจริง 🧳 หรือจะยกเลิกการจอง 🙅 มาลองทายกันเลย!</p></div></div>""",
            unsafe_allow_html=True)

left, right = st.columns([1.5, 1], gap="large")

with left:
    with st.form("predict", border=False):
        section("s1", "🛏️", "ข้อมูลการเข้าพัก", "เลือกที่พัก วันที่ และราคาห้อง")
        c1, c2 = st.columns(2)
        hotel = c1.selectbox("🏠 ประเภทที่พัก", list(HOTEL_TH), format_func=HOTEL_TH.get)
        arrival = c2.date_input("📅 วันที่เข้าพัก", dt.date.today() + dt.timedelta(days=30), format="DD/MM/YYYY")
        c1, c2 = st.columns(2)
        nights = c1.number_input("🌙 จำนวนคืนที่เข้าพัก", 1, 14, 3)
        price_thb = c2.number_input("💰 ราคาห้องต่อคืน (บาท)", 1000, 15000, 3500, step=100,
                                    help="ราคาห้องจริงในข้อมูลอยู่ระหว่างประมาณ 1,000–15,000 บาท/คืน")
        st.markdown('<div class="gap"></div>', unsafe_allow_html=True)

        section("s2", "📝", "ข้อมูลการจอง", "จองยังไง มัดจำไหม และมาจากประเทศอะไร")
        c1, c2 = st.columns(2)
        lead_time = c1.number_input("⏰ จองล่วงหน้า (วัน)", 0, 700, 60)
        deposit = c2.selectbox("💳 ประเภทเงินมัดจำ", list(DEPOSIT_TH), format_func=DEPOSIT_TH.get)
        c1, c2 = st.columns(2)
        segment = c1.selectbox("🛒 ช่องทางการจอง", list(SEGMENT_TH), format_func=SEGMENT_TH.get)
        country = c2.selectbox("🌍 ประเทศของลูกค้า", list(COUNTRY_TH),
                               format_func=lambda c: f"{COUNTRY_FLAG[c]}  {COUNTRY_TH[c]}")

        st.write("")
        submitted = st.form_submit_button("🔮  ทายเลย!", use_container_width=True)

with right:
    if not submitted:
        st.markdown("""<div class="card wait"><div class="big">🤔</div>
          <h3>ลูกค้าจะมาไหมนะ?</h3><p>กรอกข้อมูลแล้วให้ AI ช่วยทาย</p>
          <div class="steps">
            <div class="step st1"><i>✏️</i>กรอกข้อมูลการจองทางซ้าย</div>
            <div class="step st2"><i>👆</i>กดปุ่ม <b>&nbsp;ทายเลย!</b></div>
            <div class="step st3"><i>🎉</i>ดูผลว่าลูกค้าจะมาหรือยกเลิก</div>
          </div></div>""", unsafe_allow_html=True)
    else:
        stay_dates = [arrival + dt.timedelta(days=i) for i in range(nights)]
        weekend = sum(d.weekday() >= 5 for d in stay_dates)
        p_base, contrib = predict(model, dict(
            hotel=hotel, market_segment=segment, country=country, arrival_month=arrival.month,
            lead_time=lead_time, nights=nights, weekend_nights=weekend,
            adr=price_thb / EUR_TO_THB))                    # บาท -> ยูโร ก่อนเข้าโมเดล
        p = adjust_for_deposit(p_base, deposit)
        reasons = explain(contrib, p_base, p, deposit, lead_time, nights, price_thb, hotel, segment,
                          country, arrival)
        cancel = p >= 0.5
        if cancel:
            face, txt, color, sub = "😢", "ยกเลิก", "#ff4d6d", "โอ๊ะโอ! ลูกค้าคนนี้น่าจะยกเลิกการจอง"
            bar = "repeating-linear-gradient(45deg,#ff4d6d 0 14px,#ff7a90 14px 28px)"
        else:
            face, txt, color, sub = "🥳", "ไม่ยกเลิก", "#12b76a", "เย้! ลูกค้าคนนี้น่าจะมาพักจริง"
            bar = "repeating-linear-gradient(45deg,#3ddc97 0 14px,#6ee7b7 14px 28px)"
        checkout = arrival + dt.timedelta(days=nights)
        st.markdown(f"""<div class="card">
          <div class="face">{face}</div>
          <div class="verdict" style="color:{color}">{txt}</div>
          <div class="vsub">{sub}</div>
          <div style="text-align:left;color:#3b2a5c;font-weight:500">🎯 โอกาสยกเลิก</div>
          <div class="meter"><div style="width:{max(p, 0.001) * 100:.1f}%;background:{bar}">{p:.0%}</div></div>
          <div class="scale"><span>😊 มาแน่</span><span>🤷 ครึ่งๆ</span><span>😢 ยกเลิกแน่</span></div>
          <div class="why"><div class="why-t">💡 ทำไมถึงทายแบบนี้?</div>{reasons}</div>
          <div class="chips">
            <span class="chip">{HOTEL_TH[hotel]}</span>
            <span class="chip">{COUNTRY_FLAG[country]} {COUNTRY_TH[country]}</span>
            <span class="chip">📅 {thai_date(arrival)} – {thai_date(checkout)}</span>
            <span class="chip">{SEGMENT_TH[segment]}</span>
            <span class="chip">{DEPOSIT_TH[deposit]}</span>
          </div>
          <div class="money"><span>💰 มูลค่าการจอง<br>{nights} คืน × ฿{price_thb:,}</span>
            <b>฿{price_thb * nights:,}</b></div>
        </div>""", unsafe_allow_html=True)
        if not cancel:
            st.balloons()

st.markdown('<div class="foot">🏨 ทายใจนักท่องเที่ยว · ระบบทำนายการยกเลิกการจองห้องพักด้วย AI</div>',
            unsafe_allow_html=True)