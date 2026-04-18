import streamlit as st
import requests
from PIL import Image
import io
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🚗",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    .approve-badge {
        background-color: #1a7a1a;
        color: white;
        padding: 16px 32px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }
    .review-badge {
        background-color: #a85e00;
        color: white;
        padding: 16px 32px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }
    .reject-badge {
        background-color: #7a1a1a;
        color: white;
        padding: 16px 32px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
    }
    .info-box {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .reason-item {
        background-color: #2a2d3e;
        border-left: 4px solid #e05252;
        padding: 10px 14px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

FASTAPI_URL = "http://localhost:8000/claim"

# ── Header ────────────────────────────────────────────────────
st.title("🚗 Insurance Fraud Detection System")
st.caption("Upload a vehicle image and describe the damage claim to get an instant fraud assessment.")
st.divider()

# ── Input Section ─────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📷 Vehicle Image")
    uploaded_image = st.file_uploader(
        "Upload car photo",
        type=["jpg", "jpeg", "png"],
        help="Upload the vehicle photo submitted with the claim"
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

with col_right:
    st.subheader("📋 Claim Details")
    claim_text = st.text_area(
        "Describe the damage",
        placeholder="e.g. Car has a dent on the front bumper and a scratch on the door...",
        height=120
    )
    claim_amount = st.number_input(
        "Claim Amount (₹)",
        min_value=1000,
        max_value=1000000,
        value=50000,
        step=1000
    )
    vehicle_age = st.slider(
        "Vehicle Age (years)",
        min_value=1,
        max_value=20,
        value=5
    )

st.divider()

# ── Submit ────────────────────────────────────────────────────
submit = st.button("🔍 Analyse Claim", use_container_width=True, type="primary")

# ── Results ───────────────────────────────────────────────────
if submit:
    if not uploaded_image:
        st.warning("Please upload a vehicle image first.")
    elif not claim_text.strip():
        st.warning("Please describe the damage in the claim text.")
    else:
        with st.spinner("Analysing claim — running YOLO, OCR, and fraud model..."):
            try:
                files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                data  = {
                    "claim_text":   claim_text,
                    "claim_amount": claim_amount,
                    "vehicle_age":  vehicle_age
                }
                response = requests.post(FASTAPI_URL, files=files, data=data)
                result   = response.json()

            except Exception as e:
                st.error(f"Could not connect to backend: {e}")
                st.stop()

        st.success("Analysis complete!")
        st.divider()

        # ── Decision Badge ────────────────────────────────────
        decision    = result["decision"]["decision"]
        reason      = result["decision"]["reason"]
        fraud_prob  = result["fraud"]["fraud_probability"]
        badge_class = {"APPROVE": "approve-badge",
                       "REVIEW":  "review-badge",
                       "REJECT":  "reject-badge"}.get(decision, "review-badge")

        st.markdown(f'<div class="{badge_class}">{"✅" if decision=="APPROVE" else "⚠️" if decision=="REVIEW" else "❌"} {decision}</div>', unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center; color:#aaa; margin-top:8px'>{reason}</p>", unsafe_allow_html=True)
        st.divider()

        # ── Three Column Results ──────────────────────────────
        r1, r2, r3 = st.columns(3, gap="medium")

        with r1:
            st.subheader("🎯 Fraud Probability")
            color = "#e05252" if fraud_prob > 0.7 else "#e0a050" if fraud_prob > 0.4 else "#52e07a"
            st.markdown(f"""
                <div style='text-align:center; font-size:52px; font-weight:bold; color:{color}'>
                    {round(fraud_prob * 100, 1)}%
                </div>
            """, unsafe_allow_html=True)
            st.progress(fraud_prob)

        with r2:
            st.subheader("🔍 YOLO Detections")
            detections = result["yolo"]["detections"]
            if detections:
                for det in detections:
                    st.markdown(f"""
                        <div class='info-box'>
                            <b>{det['label']}</b><br>
                            <span style='color:#aaa'>Confidence: {round(det['confidence']*100, 1)}%</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No damage detected by YOLO")

        with r3:
            st.subheader("🪪 OCR Plate Check")
            plate_text  = result["ocr"]["plate_text"]
            plate_valid = result["ocr"]["plate_valid"]
            st.markdown(f"""
                <div class='info-box'>
                    <b>Plate Found:</b> {plate_text if plate_text else "Not detected"}<br>
                    <b>Status:</b> {"✅ Valid" if plate_valid else "❌ Invalid / Not in database"}
                </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Match Score")
            match_score = result["match"]["match_score"]
            match_color = "#52e07a" if match_score > 0.6 else "#e0a050" if match_score > 0.3 else "#e05252"
            st.markdown(f"""
                <div style='font-size:36px; font-weight:bold; color:{match_color}; text-align:center'>
                    {round(match_score * 100, 1)}%
                </div>
            """, unsafe_allow_html=True)
            st.caption("How much the claim matches YOLO detections")

        st.divider()

        # ── XAI Reasons ───────────────────────────────────────
        st.subheader("🧠 Why This Decision Was Made")
        reasons = result["xai"]["top_reasons"]
        for i, reason_text in enumerate(reasons, 1):
            st.markdown(f"""
                <div class='reason-item'>
                    <b>Reason {i}:</b> {reason_text}
                </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Images Side by Side ───────────────────────────────
        img_col1, img_col2 = st.columns(2, gap="medium")

        with img_col1:
            st.subheader("📸 Annotated Image (YOLO Boxes)")
            annotated_path = result["yolo"]["annotated_image_path"]
            if os.path.exists(annotated_path):
                st.image(annotated_path, use_column_width=True)
            else:
                st.info("Annotated image not found")

        with img_col2:
            st.subheader("📉 SHAP Explanation")
            shap_path = "outputs/shap_explanation.png"
            if os.path.exists(shap_path):
                st.image(shap_path, use_column_width=True)
            else:
                st.info("SHAP chart not available")

        st.divider()

        # ── Raw JSON (Expandable) ─────────────────────────────
        with st.expander("🔧 View Raw API Response"):
            st.json(result)