import pickle
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="CUET UG | Cutoff Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).resolve().parent / "gradient_boosting_model.pkl"

# IMPORTANT: Keep these names and their order exactly as trained.
FEATURES = [
    "university_id",
    "university",
    "year",
    "programme_id",
    "programme",
    "college",
    "variant",
    "round",
    "category",
    "seat_type",
    "figure_kind",
    "basis",
    "source_id",
    "source_page",
]

# ============================================================
# MODERN UI + ANIMATIONS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, rgba(100,80,255,.16), transparent 32%),
        radial-gradient(ellipse at 95% 15%, rgba(40,190,210,.10), transparent 30%),
        #080c16;
    color: #f4f6ff;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #121a2d, #0b1120);
    border-right: 1px solid rgba(150,165,210,.16);
}

[data-testid="stSidebar"] * {
    color: #d8def0;
}

.topbar {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 15px 18px;
    margin-bottom: 22px;
    background: rgba(17,25,43,.86);
    border: 1px solid rgba(150,165,210,.19);
    border-radius: 17px;
    animation: rise .6s ease both;
}

.brand {
    width: 45px;
    height: 45px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 14px;
    background: linear-gradient(135deg,#7768ff,#43c4d5);
    color: white;
    font-family: 'Space Grotesk',sans-serif;
    font-weight: 700;
    box-shadow: 0 5px 22px rgba(111,98,255,.3);
}

.brand-text {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.brand-text strong {
    font-size: 13px;
    letter-spacing: 1.5px;
}

.brand-text span {
    color: #8491b0;
    font-size: 10px;
    letter-spacing: 1px;
}

.status {
    margin-left: auto;
    color: #92ead1;
    font-size: 10px;
    letter-spacing: 1px;
    border: 1px solid rgba(90,220,180,.25);
    background: rgba(45,170,130,.09);
    border-radius: 999px;
    padding: 9px 12px;
}

.hero {
    position: relative;
    overflow: hidden;
    padding: 34px;
    margin-bottom: 22px;
    border-radius: 25px;
    border: 1px solid rgba(170,160,255,.22);
    background:
        radial-gradient(circle at 88% 10%, rgba(120,85,255,.32), transparent 32%),
        linear-gradient(120deg,#17233b,#111827 55%,#251c49);
    box-shadow: 0 20px 60px rgba(0,0,0,.22);
    animation: rise .75s ease both;
}

.hero:after {
    content: "";
    position: absolute;
    pointer-events: none;
    top: -100%;
    left: -40%;
    width: 35%;
    height: 300%;
    transform: rotate(25deg);
    background: linear-gradient(90deg,transparent,rgba(255,255,255,.045),transparent);
    animation: shine 5s ease-in-out infinite;
}

.eyebrow {
    color: #b4aaff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Space Grotesk',sans-serif;
    color: white;
    font-size: clamp(32px,4vw,52px);
    line-height: 1.08;
    letter-spacing: -1.8px;
    margin: 15px 0;
}

.hero p {
    color: #c0cbe0;
    font-size: 15px;
    line-height: 1.8;
    max-width: 760px;
}

.tag {
    display: inline-block;
    margin: 5px 5px 0 0;
    padding: 7px 11px;
    color: #d5ceff;
    background: rgba(125,105,255,.12);
    border: 1px solid rgba(160,145,255,.27);
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: .4px;
}

.section-title {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #f5f6ff;
    margin: 26px 0 5px;
}

.subtext {
    color: #8f9bb6;
    font-size: 13px;
    line-height: 1.7;
    margin-bottom: 18px;
}

.metric {
    padding: 20px;
    min-height: 118px;
    border: 1px solid #293651;
    border-radius: 17px;
    background: linear-gradient(145deg,#151f34,#101625);
    box-shadow: 0 9px 28px rgba(0,0,0,.12);
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    animation: rise .6s ease both;
}

.metric:hover {
    transform: translateY(-4px);
    border-color: #7569ed;
    box-shadow: 0 15px 32px rgba(91,76,210,.16);
}

.metric-label {
    color: #93a0bc;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.metric-value {
    font-family: 'Space Grotesk',sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #fff;
    margin-top: 13px;
}

div[data-testid="stForm"] {
    padding: 24px;
    border: 1px solid #293651;
    border-radius: 20px;
    background: linear-gradient(145deg,rgba(19,28,47,.96),rgba(12,18,31,.97));
    box-shadow: 0 15px 40px rgba(0,0,0,.15);
}

label {
    color: #cbd4e8 !important;
    font-size: 12px !important;
}

div[data-baseweb="input"] input,
div[data-baseweb="select"] > div {
    background: #0a1020 !important;
    color: #f5f7ff !important;
    border-color: #303c57 !important;
    border-radius: 10px !important;
}

div[data-baseweb="input"] input:focus {
    border-color: #8a7aff !important;
    box-shadow: 0 0 0 2px rgba(138,122,255,.15) !important;
}

.stButton > button,
div[data-testid="stFormSubmitButton"] > button,
[data-testid="stDownloadButton"] > button {
    color: white;
    background: linear-gradient(105deg,#7164f5,#9667f4,#498fdf);
    background-size: 180% 180%;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
    transition: transform .2s ease, box-shadow .2s ease, background-position .3s ease;
    box-shadow: 0 7px 20px rgba(105,89,240,.2);
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    color: white;
    transform: translateY(-2px);
    background-position: 100% 50%;
    box-shadow: 0 11px 27px rgba(105,89,240,.3);
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #293651;
    border-radius: 13px;
    overflow: hidden;
}

.footer {
    text-align: center;
    padding-top: 22px;
    margin-top: 35px;
    border-top: 1px solid #222d43;
    color: #7885a1;
    font-size: 11px;
    letter-spacing: .4px;
}

@keyframes rise {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes shine {
    0%, 20% { left: -50%; }
    60%, 100% { left: 140%; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }
}

@media(max-width:700px) {
    .block-container {
        padding: 1rem .8rem 2rem;
    }
    .hero {
        padding: 23px;
        border-radius: 19px;
    }
    .hero h1 {
        font-size: 34px;
        letter-spacing: -1px;
    }
    .topbar {
        padding: 11px;
    }
    .status {
        font-size: 8px;
        padding: 7px;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL LOADING
# ============================================================
@st.cache_resource
def load_model_from_path(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@st.cache_resource
def load_model_from_bytes(model_bytes):
    return pickle.loads(model_bytes)


# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="topbar">
    <div class="brand">CU·AI</div>
    <div class="brand-text">
        <strong>CUTOFF INTELLIGENCE</strong>
        <span>CUET UG · ADMISSIONS PLANNING WORKSPACE</span>
    </div>
    <div class="status">● PREDICTION LAB</div>
</div>

<div class="hero">
    <div class="eyebrow">✦ CUET UG · FUTURE ADMISSIONS</div>
    <h1>Plan your future.<br>Explore your possibilities.</h1>
    <p>
        Estimate upcoming CUET UG cutoffs for 2027, 2028, 2029 and 2030.
        Select your university, programme and category to compare model-generated
        estimates across future admission years.
    </p>
    <span class="tag">GRADIENT BOOSTING REGRESSOR</span>
    <span class="tag">2027 — 2030 FORECAST</span>
    <span class="tag">DOWNLOADABLE RESULTS</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("## 🎓 CUET UG")
st.sidebar.caption("CUTOFF INTELLIGENCE")
st.sidebar.divider()

st.sidebar.markdown("### How to use")
st.sidebar.markdown("""
1. Load the trained model.
2. Select your forecast years.
3. Enter university, programme and category codes.
4. Click **Generate forecast**.
5. Review the graph and download the results.
""")

st.sidebar.info(
    "This model requires the same numeric encodings used during training. "
    "The pickle file does not contain the original category mappings."
)

st.sidebar.caption("Estimates only · Not official admission advice")


# ============================================================
# LOAD MODEL OR REQUEST UPLOAD
# ============================================================
model = None

if MODEL_PATH.is_file():
    try:
        model = load_model_from_path(str(MODEL_PATH))
        st.sidebar.success("Trained model loaded from repository.")
    except Exception as exc:
        st.error(f"Unable to load the repository model: {exc}")

if model is None:
    st.warning(
        "The trained model file is missing. Upload your original "
        "`gradient_boosting_model.pkl` below, or add it to the GitHub "
        "repository root and redeploy."
    )

    uploaded_model = st.file_uploader(
        "Upload gradient_boosting_model.pkl",
        type=["pkl"],
        help="Upload only your trusted, original trained model file."
    )

    if uploaded_model is not None:
        try:
            model = load_model_from_bytes(uploaded_model.getvalue())
            st.sidebar.success("Uploaded model loaded.")
        except Exception as exc:
            st.error(f"Could not load the uploaded model: {exc}")
            st.stop()
    else:
        st.info(
            "The forecast form will appear after a valid model has been loaded. "
            "Pickle files can execute code when opened, so upload only a file you trust."
        )
        st.stop()


# ============================================================
# STATUS CARDS
# ============================================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Model status</div>
        <div class="metric-value">● Ready</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Forecast window</div>
        <div class="metric-value">2027–2030</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
        <div class="metric-label">Model inputs</div>
        <div class="metric-value">14 features</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FORECAST INPUT FORM
# ============================================================
st.markdown(
    '<div class="section-title">Build your cutoff forecast</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtext">Choose your admission scenario once. '
    'The model will evaluate each selected future year using the same other inputs.</div>',
    unsafe_allow_html=True
)

with st.form("forecast_form"):
    st.markdown("#### 01 · Forecast window")

    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        start_year = st.selectbox(
            "First admission year",
            options=[2027, 2028, 2029, 2030],
            index=0
        )

    with fc2:
        end_year = st.selectbox(
            "Last admission year",
            options=[2027, 2028, 2029, 2030],
            index=3
        )

    with fc3:
        round_num = st.number_input(
            "Counselling round",
            min_value=0,
            max_value=50,
            value=1,
            step=1
        )

    st.markdown("#### 02 · University and programme")

    a, b, c = st.columns(3)

    with a:
        university_id = st.number_input(
            "University ID code", min_value=0, value=0, step=1
        )
        university = st.number_input(
            "University encoded value", value=0.0, step=1.0
        )

    with b:
        programme_id = st.number_input(
            "Programme ID code", min_value=0, value=0, step=1
        )
        programme = st.number_input(
            "Programme encoded value", value=0.0, step=1.0
        )

    with c:
        college = st.number_input(
            "College encoded value", value=0.0, step=1.0
        )
        variant = st.number_input(
            "Programme variant code", value=0.0, step=1.0
        )

    st.markdown("#### 03 · Category and seat details")

    d, e, f = st.columns(3)

    with d:
        category = st.number_input(
            "Category code", value=0.0, step=1.0
        )
        seat_type = st.number_input(
            "Seat type code", value=0.0, step=1.0
        )

    with e:
        figure_kind = st.number_input(
            "Cutoff figure type code", value=0.0, step=1.0
        )
        basis = st.number_input(
            "Cutoff basis code", value=0.0, step=1.0
        )

    st.markdown("#### 04 · Source information")

    g, h = st.columns(2)

    with g:
        source_id = st.number_input(
            "Source ID", min_value=0, value=0, step=1
        )

    with h:
        source_page = st.number_input(
            "Source page", min_value=0, value=0, step=1
        )

    submitted = st.form_submit_button(
        "✦ Generate future cutoff forecast",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================
if submitted:
    if end_year < start_year:
        st.error("The last admission year must be the same as or later than the first year.")
        st.stop()

    years = list(range(int(start_year), int(end_year) + 1))

    base_values = [
        university_id,
        university,
        start_year,  # replaced with each forecast year below
        programme_id,
        programme,
        college,
        variant,
        round_num,
        category,
        seat_type,
        figure_kind,
        basis,
        source_id,
        source_page,
    ]

    forecast_rows = []

    for target_year in years:
        row = list(base_values)
        row[FEATURES.index("year")] = target_year
        forecast_rows.append(row)

    forecast_df = pd.DataFrame(forecast_rows, columns=FEATURES)

    try:
        predictions = model.predict(forecast_df)

        results = pd.DataFrame({
            "Admission Year": years,
            "Estimated Cutoff": [float(p) for p in predictions],
        })

        st.divider()
        st.markdown(
            '<div class="section-title">Your future cutoff estimates</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="subtext">Forecast for {start_year}–{end_year}. '
            'Only the admission year changes between predictions.</div>',
            unsafe_allow_html=True
        )

        # Main result
        first_prediction = float(predictions[0])
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(f"""
            <div class="metric">
                <div class="metric-label">First forecast · {years[0]}</div>
                <div class="metric-value">{first_prediction:,.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            last_prediction = float(predictions[-1])
            st.markdown(f"""
            <div class="metric">
                <div class="metric-label">Last forecast · {years[-1]}</div>
                <div class="metric-value">{last_prediction:,.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            difference = last_prediction - first_prediction
            st.markdown(f"""
            <div class="metric">
                <div class="metric-label">Model output difference</div>
                <div class="metric-value">{difference:+,.3f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Cutoff trend across years")

        chart_df = results.set_index("Admission Year")
        st.line_chart(
            chart_df,
            use_container_width=True,
            height=350
        )

        st.markdown("#### Forecast table")

        display_results = results.copy()
        display_results["Estimated Cutoff"] = display_results[
            "Estimated Cutoff"
        ].map(lambda value: f"{value:,.3f}")

        st.dataframe(
            display_results,
            use_container_width=True,
            hide_index=True
        )

        csv_bytes = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇ Download forecast as CSV",
            data=csv_bytes,
            file_name=f"cuet_cutoff_forecast_{start_year}_{end_year}.csv",
            mime="text/csv",
            use_container_width=True
        )

        with st.expander("View exact model inputs"):
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "These are model-generated estimates, not official CUET UG cutoffs. "
            "A Gradient Boosting model does not reliably extrapolate long-term trends "
            "beyond its training data. Estimates for future years may be flat or inaccurate. "
            "Verify admission decisions against official university announcements."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.info(
            "Check that the model is the original trained model and that all 14 "
            "features use the same names, order, data types and encodings as training."
        )


# ============================================================
# MODEL INFORMATION
# ============================================================
with st.expander("About the model and forecast"):
    st.markdown("""
    **Algorithm:** Gradient Boosting Regressor

    **Input features:** 14 numeric features, supplied in the exact order
    used by the original model.

    **Future-year method:** The app runs the trained model repeatedly,
    changing only the admission-year input. It is not a separately trained
    time-series forecasting model.

    **Important limitation:** The model's pickle file does not include the
    original category mappings or historical training dataset. Input codes
    must match the original preprocessing exactly. Zero is not necessarily
    a valid code for every field.
    """)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        st.markdown("#### Feature importance")
        st.bar_chart(
            importance.set_index("Feature"),
            use_container_width=True
        )


st.markdown("""
<div class="footer">
    CUET UG CUTOFF INTELLIGENCE · STREAMLIT ·
    MODEL ESTIMATES ONLY · NOT OFFICIAL ADMISSION ADVICE
</div>
""", unsafe_allow_html=True)
```
