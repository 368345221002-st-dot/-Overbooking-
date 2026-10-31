# App/V5 — ระบบเว็บต้นแบบของบทความ 002-030

บทความ: การจัดระดับความเสี่ยงการยกเลิกการจองจากตัวแปรที่ทราบ ณ เวลาจองเพื่อกำหนดจำนวนห้องรับจองเกินของโรงแรม
(Cancellation Risk Levels from Booking-Time Features for Setting Hotel Overbooking Quantities)

## ไฟล์ในโฟลเดอร์นี้

| ไฟล์ | หน้าที่ |
|---|---|
| `app.py` | ระบบเว็บ Streamlit รับข้อมูลการจอง 7 ช่อง แล้วแสดงความน่าจะเป็นที่จะยกเลิกและระดับความเสี่ยง (ต่ำ ≤ 0.40 < ปานกลาง < 0.60 ≤ สูง) |
| `hgb_web5_pipeline.joblib` | ตัวแบบ HGB บนชุดฟีเจอร์ WEB5 md5 `b53adb8480eb6ec695406149f6d87556` **ไฟล์เดียวกับ `Code/V5/results/hgb_web5_pipeline.joblib`** |
| `requirements.txt` | รุ่นไลบรารี: streamlit 1.50.0, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, joblib 1.6.0 |
| `Link.pdf` | ลิงก์ระบบออนไลน์ |
| `screenshot_online_V5.png` | ภาพหน้าจอระบบออนไลน์ที่เห็น URL |

## ตรวจว่าแอปใช้ตัวแบบเดียวกับการทดลอง
- md5 ของ `hgb_web5_pipeline.joblib` ในโฟลเดอร์นี้ = md5 ของไฟล์ใน `Code/V5/results/` = `b53adb8480eb…`
- ตัวอย่างในรูปที่ 4: จองล่วงหน้า 200 วัน, ADR 100, ผู้ใหญ่ 2 คน, ห้องสแตนดาร์ด (A), ประเทศโปรตุเกส (PRT) → โอกาสยกเลิก **83.2%** (ระดับสูง)
  ตรงกับ `Code/V5/results/results_test.json` → `web_example_figure4` → `p_cancel` = 0.8316
- ขอบบน ADR ใน app.py (`ADR_CAP = 227.388`) ตรงกับ `results_data.json` → `data → adr_upper_iqr_train`

## วิธีรัน
```
pip install -r requirements.txt
streamlit run app.py
```

## ประวัติการแก้ไข
- **V5** (แก้ตามรายงานผลการพิจารณา V4 ข้อ 3.5): สร้างโฟลเดอร์ App/V5 โดยคัดลอก `app.py`, `requirements.txt`, `Link.pdf` จาก App/V3
  และคัดลอก `hgb_web5_pipeline.joblib` จาก `Code/V5/results/` แก้ใน app.py เฉพาะข้อความหัวไฟล์และหมายเหตุ (V3 → V5, อ้างชื่อหัวข้อแทนเลขหัวข้อ)
  **ไม่ได้แก้การทำงานของแอป**
