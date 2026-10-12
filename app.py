import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)
st.set_page_config(page_title="CUET UG | Cutoff Intelligence", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
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

/* Polished visual layer: subtle motion, glass surfaces and responsive spacing */

/* V2 redesign: clearly distinct header, visual hierarchy and polished surfaces */
.topline { display:flex; align-items:center; gap:13px; margin:0 0 20px; padding:13px 16px; border:1px solid rgba(153,167,207,.17); border-radius:16px; background:rgba(14,21,36,.78); box-shadow:0 10px 28px rgba(0,0,0,.12); animation:heroRise .5s ease both; }
.brand-mark { display:flex; align-items:center; justify-content:center; width:42px; height:42px; flex:0 0 42px; border-radius:13px; color:#fff; font-family:'Space Grotesk',sans-serif; font-size:15px; font-weight:700; background:linear-gradient(135deg,#7667ff,#42bfd1); box-shadow:0 5px 18px rgba(110,99,255,.3); }
.brand-mark span { color:#c9c2ff; margin:0 2px; }
.topline-copy { display:flex; flex-direction:column; gap:3px; }
.topline-copy strong { font-size:12px; letter-spacing:1.5px; color:#f5f6ff; }
.topline-copy span { font-size:9px; letter-spacing:1.25px; color:#8593b1; }
.live-pill { margin-left:auto; display:flex; align-items:center; gap:8px; border:1px solid rgba(110,207,188,.22); border-radius:999px; padding:8px 11px; color:#b5eadf; background:rgba(36,139,119,.08); font-size:9px; font-weight:700; letter-spacing:.9px; white-space:nowrap; }
.live-dot { width:7px; height:7px; border-radius:50%; background:#56dfb8; box-shadow:0 0 0 4px rgba(86,223,184,.12); animation:livePulse 2s ease infinite; }
.version-tag { border-left:1px solid rgba(181,234,223,.25); padding-left:8px; color:#c4baff; }
@keyframes livePulse { 0%,100% { box-shadow:0 0 0 3px rgba(86,223,184,.10); } 50% { box-shadow:0 0 0 6px rgba(86,223,184,.02); } }
@media(max-width:650px) { .topline { gap:9px; padding:10px; } .topline-copy span { display:none; } .live-pill { font-size:8px; padding:7px; gap:5px; } .version-tag { display:none; } }

:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse at 12% 0%, rgba(92, 76, 220, .14), transparent 34%),
    radial-gradient(ellipse at 92% 18%, rgba(34, 183, 201, .08), transparent 28%),
    #080c15;
}
.block-container { max-width: 1440px; padding-top: 2.1rem; padding-bottom: 2.5rem; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#11182a 0%,#0b1120 100%); border-right:1px solid rgba(145,160,200,.18); }
[data-testid="stSidebar"] h2 { letter-spacing:-.5px; }
.hero {
  position:relative; overflow:hidden;
  background:linear-gradient(115deg,rgba(24,36,61,.98),rgba(15,23,41,.97) 52%,rgba(42,31,83,.96));
  box-shadow:0 18px 55px rgba(0,0,0,.20), inset 0 1px 0 rgba(255,255,255,.06);
  animation:heroRise .75s cubic-bezier(.2,.7,.2,1) both;
}
.hero:before {
  content:""; position:absolute; width:340px; height:340px; right:-95px; top:-175px;
  border-radius:50%; border:1px solid rgba(180,168,255,.18);
  box-shadow:0 0 0 28px rgba(180,168,255,.035),0 0 0 58px rgba(180,168,255,.025);
  animation:orbFloat 9s ease-in-out infinite alternate; pointer-events:none;
}
.hero:after {
  content:""; position:absolute; inset:0; pointer-events:none;
  background:linear-gradient(110deg,transparent 15%,rgba(255,255,255,.035) 48%,transparent 75%);
  transform:translateX(-100%); animation:heroShimmer 1.4s .25s ease-out both;
}
.hero h1 { letter-spacing:-1.8px; text-shadow:0 3px 24px rgba(122,105,255,.12); }
.eyebrow { display:inline-flex; align-items:center; gap:8px; }
.pill { backdrop-filter:blur(8px); transition:transform .22s ease, background .22s ease, border-color .22s ease; }
.pill:hover { transform:translateY(-2px); background:#302b59; border-color:#8175d9; }
.metric {
  position:relative; overflow:hidden;
  background:linear-gradient(145deg,rgba(22,32,52,.96),rgba(14,21,36,.96));
  box-shadow:0 8px 24px rgba(0,0,0,.12);
  transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease;
  animation:cardEnter .55s ease both;
}
.metric:hover { transform:translateY(-4px); border-color:#655bd0; box-shadow:0 14px 32px rgba(76,65,170,.16); }
.metric:after { content:""; position:absolute; left:0; top:0; height:2px; width:100%; background:linear-gradient(90deg,#7b6cff,#49c8d8,transparent); opacity:.75; }
div[data-testid="stForm"] {
  background:linear-gradient(145deg,rgba(17,25,42,.94),rgba(12,18,31,.97));
  box-shadow:0 18px 44px rgba(0,0,0,.15); backdrop-filter:blur(12px);
}
.stButton > button, div[data-testid="stFormSubmitButton"] > button, [data-testid="stDownloadButton"] > button {
  background:linear-gradient(105deg,#7164f5,#9667f4 58%,#5d9de8);
  background-size:180% 180%; color:#fff; border:1px solid rgba(255,255,255,.1);
  border-radius:13px; font-weight:700; min-height:46px;
  box-shadow:0 7px 20px rgba(105,89,240,.2);
  transition:transform .2s ease,box-shadow .2s ease,background-position .3s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover, [data-testid="stDownloadButton"] > button:hover {
  color:#fff; border-color:rgba(215,208,255,.65); transform:translateY(-2px);
  box-shadow:0 11px 27px rgba(105,89,240,.3); background-position:100% 50%;
}
div[data-baseweb="input"] input, div[data-baseweb="select"] > div {
  background:rgba(7,13,26,.88) !important; color:#f5f7ff !important;
  border-color:#303b53 !important; border-radius:10px !important;
  transition:border-color .2s ease,box-shadow .2s ease;
}
div[data-baseweb="input"] input:focus { border-color:#8175ff !important; box-shadow:0 0 0 2px rgba(129,117,255,.14) !important; }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border:1px solid rgba(145,160,200,.18); border-radius:14px; overflow:hidden;
}
[data-testid="stAlert"] { border-radius:12px; }
h2, h3 { letter-spacing:-.6px; }
.footer { text-align:center; letter-spacing:.25px; }
@keyframes heroRise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@keyframes cardEnter { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
@keyframes orbFloat { from { transform:translate3d(0,0,0) rotate(0deg); } to { transform:translate3d(-16px,13px,0) rotate(12deg); } }
@keyframes heroShimmer { 0% { transform:translateX(-100%); } 100% { transform:translateX(100%); } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; scroll-behavior:auto !important; }
}
@media (max-width: 700px) {
  .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
  .hero { padding:22px !important; border-radius:18px; }
  .hero h1 { letter-spacing:-1px; }
}

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
<div class="topline">
  <div class="brand-mark">CU<span>·</span>AI</div>
  <div class="topline-copy"><strong>CUTOFF INTELLIGENCE</strong><span>ADMISSIONS PLANNING WORKSPACE</span></div>
  <div class="live-pill"><span class="live-dot"></span> PREDICTION LAB <span class="version-tag">UI v2.0</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ &nbsp; CUET UG · Admissions Intelligence</div>
  <h1>Find your next<br>academic possibility.</h1>
  <p>Explore a model-estimated CUET UG cutoff using university, programme, admission round and category-related inputs. This is a prediction from your saved regression model—not an official cutoff or admission guarantee.</p>
  <br><span class="pill">✦ GRADIENT BOOSTING REGRESSOR</span>
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

st.markdown("### Forecast CUET UG cutoffs · 2027–2030")
st.caption("Set your university, programme, category and counselling details once. The app estimates each upcoming admission year automatically. Numeric codes must match the original training encodings.")

with st.form("cuet_cutoff_form"):
    st.markdown("#### 1 · Choose your forecast window and admission details")
    col1, col2, col3 = st.columns(3)
    with col1:
        year = st.number_input("First forecast year", min_value=2027, max_value=2100, value=2027, step=1, help="Choose the first future admission year you want to estimate.")
        forecast_horizon = st.number_input("Number of future years", min_value=1, max_value=10, value=4, step=1, help="Start at 2027 and choose 4 to estimate 2027, 2028, 2029 and 2030.")
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
    st.markdown("#### 3 · Dataset source fields (advanced)")
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
        forecast_years = [int(year) + offset for offset in range(0, int(forecast_horizon))]
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
            "Forecast": ["Future-year estimate" for _ in forecast_years],
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
            st.markdown("#### Scenario details used")
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True, height=420)

        st.markdown("### Your future cutoff estimates")
        st.caption(f"Forecast window: {int(year)}–{int(year) + int(forecast_horizon) - 1}. Other scenario details stay constant while the admission year changes.")
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
            file_name=f"cuet_cutoff_forecast_{int(year)}_{int(year) + int(forecast_horizon) - 1}.csv",
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
