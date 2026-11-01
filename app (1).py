import os
import joblib
import pandas as pd
import streamlit as st

# ---------- Page configuration ----------
st.set_page_config(
    page_title="Bankruptcy Prevention | Risk Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bankruptcy_model.pkl")

# ---------- Professional styling ----------
st.markdown("""
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 28px 30px;
        border-radius: 16px;
        background: linear-gradient(120deg, #102a43, #245b78);
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 {
        color: white;
        font-size: 32px;
        margin-bottom: 8px;
    }
    .hero p {
        color: #e4edf5;
        font-size: 15px;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 21px;
        font-weight: 700;
        color: #16324f;
        margin-top: 12px;
        margin-bottom: 4px;
    }
    .help-text {
        color: #667788;
        font-size: 14px;
        margin-bottom: 18px;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 9px;
        min-height: 46px;
        font-weight: 600;
        background-color: #176b87;
        color: white;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #124e66;
        color: white;
    }
    [data-testid="stMetric"] {
        background: #f3f7fa;
        border: 1px solid #e1e9ef;
        padding: 16px;
        border-radius: 12px;
    }
    .note {
        padding: 13px 16px;
        background: #f3f7fa;
        border-left: 4px solid #176b87;
        border-radius: 6px;
        color: #34495e;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>📊 Bankruptcy Prevention System</h1>
    <p>Company risk assessment and machine-learning-based prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------- Load model ----------
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.info("Check that bankruptcy_model.pkl is in the same folder as app.py.")
    st.stop()

# ---------- Input section ----------
st.markdown('<div class="section-title">Company Risk Assessment</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="help-text">Select the risk level for each indicator, '
    'then run the prediction.</div>',
    unsafe_allow_html=True
)

risk_options = [0.0, 0.5, 1.0]

with st.form("risk_assessment"):
    col1, col2, col3 = st.columns(3)

    with col1:
        industrial_risk = st.selectbox("Industrial Risk", risk_options)
        management_risk = st.selectbox("Management Risk", risk_options)

    with col2:
        financial_flexibility = st.selectbox(
            "Financial Flexibility", risk_options
        )
        credibility = st.selectbox("Credibility", risk_options)

    with col3:
        competitiveness = st.selectbox("Competitiveness", risk_options)
        operating_risk = st.selectbox("Operating Risk", risk_options)

    submitted = st.form_submit_button("Analyze Company Risk")

# ---------- Prediction ----------
if submitted:
    input_data = pd.DataFrame([{
        "industrial_risk": industrial_risk,
        "management_risk": management_risk,
        "financial_flexibility": financial_flexibility,
        "credibility": credibility,
        "competitiveness": competitiveness,
        "operating_risk": operating_risk
    }])

    try:
        prediction = model.predict(input_data)[0]
        label = str(prediction).strip().lower()

        st.divider()
        st.markdown('<div class="section-title">Analysis Result</div>',
                    unsafe_allow_html=True)

        if label == "bankruptcy":
            st.error("### Prediction: Bankruptcy")
            st.write(
                "The trained model classified this combination of "
                "risk indicators as **Bankruptcy**."
            )
        elif label == "non-bankruptcy":
            st.success("### Prediction: Non-Bankruptcy")
            st.write(
                "The trained model classified this combination of "
                "risk indicators as **Non-Bankruptcy**."
            )
        else:
            st.warning(f"Model returned an unexpected class: {prediction}")

        st.markdown("#### Submitted Risk Indicators")
        st.dataframe(input_data, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="note">
        <b>Important:</b> This result is a machine-learning prediction
        for an academic project, not a financial audit or a guarantee
        of a company's actual financial condition.
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Bankruptcy Prevention System | Data Science Classification Project")
