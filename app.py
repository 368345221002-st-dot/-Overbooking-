"""
ระบบเว็บทำนายความเสี่ยงการยกเลิกการจองโรงแรม (ฉบับ V5)

โหลด Pipeline ของตัวแบบ WEB5 + HGB ที่บันทึกจากโน้ตบุ๊กการทดลอง
ซึ่งเป็นไฟล์เดียวกับที่ใช้รายงานผลในบทความ (การพัฒนาระบบเว็บต้นแบบ และรูปที่ 4) คัดลอกจาก Code/V5/results/
(md5 b53adb8480eb… ตัวอย่าง lead_time=200, adr=100, ผู้ใหญ่ 2 คน, ห้อง A, PRT ได้ 83.2%)

ฟีเจอร์ WEB5 (ค่าดิบ ไม่ต้องสเกลเอง Pipeline จัดการให้):
    lead_time, adr, FamilySize, reserved_room_type, country
"""

import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Hotel Booking Cancellation", page_icon="🏨", layout="centered")

# ---------------- ค่าคงที่ ----------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "hgb_web5_pipeline.joblib"
MODEL_PATH = os.path.join(APP_DIR, MODEL_NAME)

# ขอบบน ADR ที่คำนวณจากชุดฝึก (การเตรียมข้อมูล) — ต้องตรงกับคีย์ data → adr_upper_iqr_train ใน results_data.json
ADR_CAP = 227.388

# เกณฑ์ระดับความเสี่ยงเชิงปฏิบัติ (การแบ่งระดับความเสี่ยง 3 ระดับ)
T_LOW, T_HIGH = 0.40, 0.60

# ชื่อคอลัมน์ที่ Pipeline ต้องได้รับ (ต้องตรงกับตอนฝึก)
WEB5_FEATURES = ["lead_time", "adr", "FamilySize", "reserved_room_type", "country"]

# ประเภทห้อง (ชื่อไทย -> รหัสห้องในข้อมูล) เรียงตามระดับราคาเฉลี่ยจากข้อมูล
ROOM_TYPES = {
    "ห้องสแตนดาร์ด": "A",
    "ห้องซูพีเรีย": "D",
    "ห้องดีลักซ์": "F",
    "ห้องสวีท": "H",
}

# 10 ประเทศที่พบมากที่สุดในข้อมูล (ชื่อไทย -> รหัส ISO)
# "อื่น ๆ" ส่งรหัสที่ไม่อยู่ใน 10 อันดับ Pipeline จะจัดเข้าหมวด "อื่น ๆ" เอง
OTHER_LABEL = "อื่น ๆ (Other)"
OTHER_CODE = "OTHER"
COUNTRIES = {
    "โปรตุเกส": "PRT",
    "สหราชอาณาจักร": "GBR",
    "ฝรั่งเศส": "FRA",
    "สเปน": "ESP",
    "เยอรมนี": "DEU",
    "อิตาลี": "ITA",
    "ไอร์แลนด์": "IRL",
    "เบลเยียม": "BEL",
    "บราซิล": "BRA",
    "เนเธอร์แลนด์": "NLD",
    OTHER_LABEL: OTHER_CODE,
}


# ---------------- โหลดตัวแบบ ----------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.isfile(MODEL_PATH):
    st.error(f"ไม่พบไฟล์ตัวแบบ {MODEL_NAME} — วางไว้โฟลเดอร์เดียวกับ app.py")
    st.stop()

model = load_model(MODEL_PATH)

# ตรวจว่าไฟล์ที่โหลดเป็น Pipeline ที่รับฟีเจอร์ WEB5 จริง ไม่ใช่ตัวแบบรุ่นเก่า
model_features = list(getattr(model, "feature_names_in_", []))
if sorted(model_features) != sorted(WEB5_FEATURES):
    st.error(
        "ไฟล์ตัวแบบไม่ใช่ Pipeline WEB5 — คอลัมน์ที่ตัวแบบต้องการคือ "
        f"{model_features} แต่ระบบส่ง {WEB5_FEATURES}"
    )
    st.stop()

CANCEL_INDEX = list(model.classes_).index(1)


def risk_level(p):
    """แบ่งระดับความเสี่ยงตามเกณฑ์ในหัวข้อการแบ่งระดับความเสี่ยง 3 ระดับ"""
    if p >= T_HIGH:
        return "cancel", "🔴", "ความเสี่ยงสูง", "ควรติดต่อยืนยันการจอง หรือพิจารณาในการวางแผนรับจองเกิน"
    if p <= T_LOW:
        return "keep", "🟢", "ความเสี่ยงต่ำ", "มีแนวโน้มเข้าพักตามที่จอง"
    return "warn", "🟡", "ความเสี่ยงปานกลาง", "ควรติดตามการจองนี้ใกล้ชิด"


# ---------------- สไตล์ ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, label, input {
    font-family: 'Prompt', sans-serif !important;
}
.block-container { padding-top: 3rem; max-width: 760px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #38bdf8 100%);
    border-radius: 20px; padding: 28px 32px; color: #fff; margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(37, 99, 235, .25);
}
.hero h1 { color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0; padding: 0; }
.hero p  { color: #dbeafe; margin: 6px 0 0; font-weight: 300; }

.section { font-weight: 600; font-size: 1.05rem; margin: 6px 0 2px; }

[data-testid="stForm"] {
    border: 1px solid rgba(148, 163, 184, .35); border-radius: 18px;
    padding: 22px 24px 14px; box-shadow: 0 4px 18px rgba(15, 23, 42, .06);
}
.stFormSubmitButton button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9); color: #fff; border: 0;
    border-radius: 12px; padding: .7rem 0; font-size: 1.05rem; font-weight: 600;
}
.stFormSubmitButton button:hover { filter: brightness(1.08); color: #fff; }

.result {
    border-radius: 18px; padding: 24px 28px; margin-top: 24px; color: #fff;
    display: flex; align-items: center; gap: 22px;
}
.result.cancel { background: linear-gradient(135deg, #b91c1c, #ef4444); }
.result.keep   { background: linear-gradient(135deg, #047857, #10b981); }
.result.warn   { background: linear-gradient(135deg, #b45309, #f59e0b); }
.result .icon  { font-size: 3rem; line-height: 1; }
.result .label { font-size: .95rem; opacity: .9; }
.result .title { font-size: 1.6rem; font-weight: 700; }
.result .advice { font-size: .95rem; margin-top: 2px; }
.bar { background: rgba(255,255,255,.3); border-radius: 99px; height: 10px; margin-top: 10px; }
.bar > div { background: #fff; height: 100%; border-radius: 99px; }
.pct { font-size: .95rem; margin-top: 6px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>🏨 ประเมินความเสี่ยงการยกเลิกการจอง</h1>
  <p>กรอกข้อมูลการจอง แล้วให้ตัวแบบประเมินโอกาสที่ลูกค้าจะยกเลิก</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- ฟอร์ม ----------------
with st.form("booking"):
    st.markdown('<div class="section">📅 ข้อมูลการจอง</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    lead_time = c1.number_input("จองล่วงหน้า (วัน)", min_value=0, max_value=737, value=30)
    adr = c2.number_input("ราคาเฉลี่ยต่อคืน (ADR)", min_value=0.0, value=100.0, step=5.0)

    st.markdown('<div class="section">👨‍👩‍👧 จำนวนผู้เข้าพัก</div>', unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    adults = c3.number_input("ผู้ใหญ่", 0, 60, 2)
    children = c4.number_input("เด็ก", 0, 10, 0)
    babies = c5.number_input("ทารก", 0, 10, 0)

    st.markdown('<div class="section">🛏️ ห้องพักและสัญชาติ</div>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    room_label = c6.selectbox("ประเภทห้อง", list(ROOM_TYPES))
    country_label = c7.selectbox("ประเทศ", list(COUNTRIES), index=0)

    submitted = st.form_submit_button("🔮 ทำนายผล", use_container_width=True)

# ---------------- ทำนาย ----------------
if submitted:
    family_size = adults + children + babies
    if family_size == 0:
        st.warning("กรุณาระบุจำนวนผู้เข้าพักอย่างน้อย 1 คน")
        st.stop()

    # จำกัด ADR ไม่เกินขอบบนจากชุดฝึก แล้วส่งค่าดิบให้ Pipeline (หัวข้อการพัฒนาระบบเว็บต้นแบบ)
    adr_used = min(float(adr), ADR_CAP)

    X = pd.DataFrame(
        [{
            "lead_time": float(lead_time),
            "adr": adr_used,
            "FamilySize": float(family_size),
            "reserved_room_type": ROOM_TYPES[room_label],
            "country": COUNTRIES[country_label],
        }]
    )[model_features]

    p_cancel = float(model.predict_proba(X)[0][CANCEL_INDEX])
    cls, icon, title, advice = risk_level(p_cancel)

    st.markdown(
        f"""
<div class="result {cls}">
  <div class="icon">{icon}</div>
  <div style="flex:1">
    <div class="label">ผลการประเมิน</div>
    <div class="title">{title}</div>
    <div class="advice">{advice}</div>
    <div class="bar"><div style="width:{p_cancel*100:.1f}%"></div></div>
    <div class="pct">โอกาสยกเลิก {p_cancel:.1%}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if adr > ADR_CAP:
        st.caption(f"ราคาเกิน {ADR_CAP:,.3f} ระบบใช้ค่า {ADR_CAP:,.3f} ในการคำนวณ (ขอบบนจากชุดฝึก)")

    st.caption(
        f"ระดับความเสี่ยง: ต่ำ (≤ {T_LOW:.2f}) · ปานกลาง · สูง (≥ {T_HIGH:.2f}) — "
        "ระดับสูงหมายถึงโอกาสยกเลิกสูงกว่าค่าเฉลี่ย ไม่ได้หมายความว่าจะยกเลิกแน่นอน"
    )
