import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="CUET UG • Cutoff Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = Path(__file__).parent / "gradient_boosting_model.pkl"

# Feature names and order must exactly match the model's training pipeline.
FEATURES = ["university_id", "university", "year", "programme_id", "programme", "college", "variant", "round", "category", "seat_type", "figure_kind", "basis", "source_id", "source_page"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background:#090d17; color:#f6f7fb; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#101625; border-right:1px solid #273149; }
.hero { background:radial-gradient(circle at 90% 10%,rgba(95,88,255,.3),transparent 34%),linear-gradient(120deg,#172139,#111827 60%,#201936); border:1px solid #303b57; border-radius:24px; padding:30px; margin-bottom:22px; }
.eyebrow { color:#a99cff; font-size:12px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.hero h1 { font-family:'Space Grotesk',sans-serif; font-size:clamp(30px,4vw,44px); line-height:1.1; margin:10px 0; color:#fff; }
.hero p { color:#bdc7da; font-size:15px; max-width:740px; margin:0; }
.pill { display:inline-block; padding:6px 10px; border-radius:999px; background:#252344; color:#d0c7ff; border:1px solid #494279; font-size:11px; font-weight:700; letter-spacing:.4px; }
.metric { background:#121a2a; border:1px solid #29354d; padding:18px; border-radius:17px; min-height:105px; }
.metric-label { color:#8e9ab3; font-size:11px; letter-spacing:1px; text-transform:uppercase; }
.metric-value { font-family:'Space Grotesk',sans-serif; color:#fff; font-size:24px; font-weight:700; margin-top:8px; }
div[data-testid="stForm"] { background:#111827; border:1px solid #29354d; border-radius:20px; padding:22px; }
.stButton > button, div[data-testid="stFormSubmitButton"] > button { background:linear-gradient(100deg,#7667ff,#a06bff); color:white; border:0; border-radius:12px; font-weight:700; min-height:45px; }
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover { color:white; border:1px solid #c7bdff; }
div[data-baseweb="input"] input, div[data-baseweb="select"] > div { background:#0b1120 !important; color:#f5f7ff !important; border-color:#303b53 !important; }
label { color:#cbd4e5 !important; font-size:13px !important; }
.footer { color:#69758d; font-size:11px; padding:22px 0 8px; border-top:1px solid #202a3d; margin-top:28px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.sidebar.markdown("## 🎓 CUET UG")
st.sidebar.caption("CUTOFF PREDICTION LAB")
st.sidebar.markdown("---")
st.sidebar.markdown("### How to use")
st.sidebar.markdown("1. Enter the admission year and round.\n2. Enter university, programme and category codes.\n3. Click **Predict cutoff**.")
st.sidebar.warning("Use the same numeric encodings as training. The pickle does not include the original category mappings.")
st.sidebar.caption("Local inference • Student-facing demo")

st.markdown("""
<div class="hero">
  <div class="eyebrow">CUET UG · Admissions Intelligence</div>
  <h1>Find your next<br>academic possibility.</h1>
  <p>Explore a model-estimated CUET UG cutoff using university, programme, admission round and category-related inputs. This is a prediction from your saved regression model—not an official cutoff or admission guarantee.</p>
  <br><span class="pill">GRADIENT BOOSTING REGRESSOR</span>
  &nbsp; <span class="pill">CUET UG</span>
  &nbsp; <span class="pill">14 MODEL FEATURES</span>
</div>
""", unsafe_allow_html=True)

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model file: {exc}")
    st.info("Ensure gradient_boosting_model.pkl is in the same folder as app.py.")
    st.stop()

feature_count = getattr(model, "n_features_in_", len(FEATURES))
a, b, c = st.columns(3)
with a:
    st.markdown('<div class="metric"><div class="metric-label">Model status</div><div class="metric-value">● Ready</div></div>', unsafe_allow_html=True)
with b:
    st.markdown(f'<div class="metric"><div class="metric-label">Expected inputs</div><div class="metric-value">{feature_count}</div></div>', unsafe_allow_html=True)
with c:
    st.markdown('<div class="metric"><div class="metric-label">Prediction type</div><div class="metric-value">Cutoff estimate</div></div>', unsafe_allow_html=True)

st.markdown("### Configure your CUET UG scenario")
st.caption("The current model expects numeric values. University, programme and category labels need the exact encodings used during training.")

with st.form("cuet_cutoff_form"):
    st.markdown("#### Admission details")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("Admission year", min_value=2000, max_value=2100, value=2026, step=1)
        round_num = st.number_input("Counselling round", min_value=0, value=1, step=1)
        university_id = st.number_input("University ID code", min_value=0, value=0, step=1)
        university = st.number_input("University encoded value", value=0.0, step=1.0)
    with col2:
        programme_id = st.number_input("Programme ID code", min_value=0, value=0, step=1)
        programme = st.number_input("Programme encoded value", value=0.0, step=1.0)
        college = st.number_input("College encoded value", value=0.0, step=1.0)
        variant = st.number_input("Programme variant code", value=0.0, step=1.0)
    with col3:
        category = st.number_input("Category code", value=0.0, step=1.0)
        seat_type = st.number_input("Seat type code", value=0.0, step=1.0)
        figure_kind = st.number_input("Cutoff figure type code", value=0.0, step=1.0)
        basis = st.number_input("Cutoff basis code", value=0.0, step=1.0)
    st.markdown("#### Dataset source fields")
    col4, col5 = st.columns(2)
    with col4:
        source_id = st.number_input("Source ID", min_value=0, value=0, step=1)
    with col5:
        source_page = st.number_input("Source page", min_value=0, value=0, step=1)
    submitted = st.form_submit_button("🎯 Predict cutoff", use_container_width=True)

if submitted:
    values = [university_id, university, year, programme_id, programme, college, variant, round_num, category, seat_type, figure_kind, basis, source_id, source_page]
    input_df = pd.DataFrame([values], columns=FEATURES)
    try:
        prediction = float(model.predict(input_df)[0])
        st.markdown("---")
        st.markdown("### Your predicted CUET UG cutoff")
        result_col, info_col = st.columns([1.05, 1])
        with result_col:
            st.markdown(
                f'<div class="hero" style="padding:26px;margin:0"><div class="eyebrow">Estimated model output</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:42px;font-weight:700;color:#fff;margin:12px 0">{prediction:,.3f}</div>'
                f'<p>Raw output from your Gradient Boosting Regressor. The unit depends on the target used during training.</p></div>',
                unsafe_allow_html=True
            )
            st.info("This is a model estimate, not an official cutoff or an admission guarantee.")
        with info_col:
            st.markdown("#### Inputs used")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True, height=420)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write("The app expects 14 numeric features in the exact training order. Check the original preprocessing and feature names.")

with st.expander("Model information and limitations"):
    st.write("**Model:** sklearn GradientBoostingRegressor")
    st.write("**Input feature order:** " + ", ".join(FEATURES))
    st.write("The output is a raw regression value. Its unit depends on the target column used during training.")
    st.write("The pickle does not include the training dataset or category mappings; input codes must match training.")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.markdown("**Feature importance**")
        st.bar_chart(importance.set_index("Feature"))

st.markdown('<div class="footer">CUET UG CUTOFF PREDICTION LAB · Built with Streamlit · Estimates are not official admission advice</div>', unsafe_allow_html=True)
