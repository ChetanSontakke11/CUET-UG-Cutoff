import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="CUET UG • Cutoff Predictor", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
MODEL_PATH = Path(__file__).resolve().parent / "gradient_boosting_model.pkl"

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
def load_model_from_path(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_model_from_bytes(model_bytes: bytes):
    # Cache by uploaded model bytes so it is not reloaded on every rerun.
    return pickle.loads(model_bytes)

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

# Load the committed model when present. If it is missing on Streamlit Cloud,
# allow the owner to upload the trusted .pkl file through the app instead of crashing.
model = None
if MODEL_PATH.is_file():
    try:
        model = load_model_from_path(str(MODEL_PATH))
        st.sidebar.success("Trained model loaded from repository.")
    except Exception as exc:
        st.error(f"The repository model file could not be loaded: {exc}")

if model is None:
    st.warning(
        "The trained model file (gradient_boosting_model.pkl) is not present in this deployment. "
        "Upload the original model file below, or add it to the GitHub repository root and redeploy."
    )
    uploaded_model = st.file_uploader(
        "Upload gradient_boosting_model.pkl",
        type=["pkl"],
        help="Choose the trusted GradientBoostingRegressor pickle that was used with this app."
    )
    if uploaded_model is not None:
        try:
            model = load_model_from_bytes(uploaded_model.getvalue())
            st.sidebar.success("Trained model loaded from uploaded file.")
        except Exception as exc:
            st.error(f"Could not load the uploaded model: {exc}")
            st.info("Make sure you selected the original, valid gradient_boosting_model.pkl file.")
            st.stop()
    else:
        st.info(
            "Predictions will become available after the model is loaded. "
            "Pickle files can execute code when opened, so upload only a model file you trust."
        )
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
        year = st.number_input("Reference admission year", min_value=2000, max_value=2100, value=2026, step=1, help="The app will predict this year and project the same scenario into upcoming years.")
        forecast_horizon = st.number_input("Upcoming years to project", min_value=1, max_value=5, value=3, step=1, help="Creates model estimates for the next 1–5 years while keeping the other inputs unchanged.")
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
    base_values = [university_id, university, year, programme_id, programme, college, variant, round_num, category, seat_type, figure_kind, basis, source_id, source_page]
    input_df = pd.DataFrame([base_values], columns=FEATURES)
    try:
        # Predict the selected reference year and each requested future year.
        forecast_years = [int(year) + offset for offset in range(0, int(forecast_horizon) + 1)]
        forecast_rows = []
        for target_year in forecast_years:
            row_values = list(base_values)
            row_values[FEATURES.index("year")] = target_year
            forecast_rows.append(row_values)

        forecast_df = pd.DataFrame(forecast_rows, columns=FEATURES)
        forecast_values = model.predict(forecast_df)
        forecast_table = pd.DataFrame({
            "Admission year": forecast_years,
            "Model-estimated cutoff": [float(value) for value in forecast_values],
            "Year type": ["Reference year" if target_year == int(year) else "Future projection"
                          for target_year in forecast_years],
        })

        reference_prediction = float(forecast_values[0])
        st.markdown("---")
        st.markdown("### CUET UG cutoff forecast")
        result_col, info_col = st.columns([1.05, 1])
        with result_col:
            st.markdown(
                f'<div class="hero" style="padding:26px;margin:0"><div class="eyebrow">Reference-year estimate · {int(year)}</div>'
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:42px;font-weight:700;color:#fff;margin:12px 0">{reference_prediction:,.3f}</div>'
                f'<p>Raw output from your Gradient Boosting Regressor. The unit depends on the target used during training.</p></div>',
                unsafe_allow_html=True
            )
        with info_col:
            st.markdown("#### Inputs used")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True, height=420)

        st.markdown("### Upcoming-year projections")
        st.caption(f"Same university, programme, category and counselling details are held constant; only the admission year changes from {int(year)} to {int(year) + int(forecast_horizon)}.")
        st.dataframe(
            forecast_table.style.format({"Model-estimated cutoff": "{:,.3f}"}),
            use_container_width=True,
            hide_index=True
        )
        chart_data = forecast_table.set_index("Admission year")[["Model-estimated cutoff"]]
        st.line_chart(chart_data, use_container_width=True)

        csv_data = forecast_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download forecast as CSV",
            data=csv_data,
            file_name=f"cuet_cutoff_forecast_{int(year)}_{int(year) + int(forecast_horizon)}.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.warning(
            "Important: these are model outputs for future-year inputs, not a separately trained time-series forecast. "
            "Gradient Boosting trees generally do not extrapolate long-term trends beyond the years represented in training; "
            "future estimates may stay flat or be unreliable. Validate against official university cutoffs and retrain with newer data."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write("The app expects 14 numeric features in the exact training order. Check the original preprocessing and feature names.")

with st.expander("Model information and limitations"):
    st.write("**Model:** sklearn GradientBoostingRegressor")
    st.write("**Input feature order:** " + ", ".join(FEATURES))
    st.write("The output is a raw regression value. Its unit depends on the target column used during training.")
    st.write("Future-year projections re-run the same model with only the year feature changed. They are not a dedicated time-series forecast; tree-based models may not extrapolate future trends reliably.")
    st.write("The pickle does not include the training dataset or category mappings; input codes must match training.")
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        st.markdown("**Feature importance**")
        st.bar_chart(importance.set_index("Feature"))

st.markdown('<div class="footer">CUET UG CUTOFF PREDICTION LAB · Built with Streamlit · Estimates are not official admission advice</div>', unsafe_allow_html=True)
