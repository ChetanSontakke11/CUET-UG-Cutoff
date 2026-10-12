
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CUET UG | Future Cutoff Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_PATH = Path(__file__).parent / "data" / "cuet_cutoffs.csv"
FUTURE_YEARS = [2027, 2028, 2029, 2030]

# =========================================================
# CUSTOM CSS / ANIMATIONS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #172554 0%, transparent 38%),
        radial-gradient(ellipse at bottom right, #312e81 0%, transparent 32%),
        #070b16;
    color: #f8fafc;
}

[data-testid="stHeader"] {
    background: rgba(7, 11, 22, 0.75);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

.hero {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(120deg, #111c3d, #25205a, #12354d);
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 15px 45px rgba(0, 0, 0, 0.22);
    animation: fadeIn 0.8s ease-out;
}

.hero h1 {
    font-size: clamp(30px, 4vw, 48px);
    line-height: 1.15;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 12px;
}

.hero p {
    color: #cbd5e1;
    font-size: 16px;
}

.gradient-text {
    background: linear-gradient(90deg, #67e8f9, #a78bfa, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.section-title {
    font-size: 23px;
    font-weight: 750;
    margin-top: 24px;
    margin-bottom: 12px;
}

.glass-card {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.20);
    padding: 20px;
    border-radius: 18px;
    transition: transform 0.25s ease, border-color 0.25s ease;
    animation: fadeIn 0.7s ease-out;
}

.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(103, 232, 249, 0.65);
}

.small-label {
    color: #94a3b8;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

.big-value {
    font-size: 27px;
    font-weight: 800;
    color: #67e8f9;
    margin-top: 7px;
}

div.stButton > button {
    border: 0;
    border-radius: 12px;
    min-height: 46px;
    font-weight: 700;
    color: white;
    background: linear-gradient(100deg, #0891b2, #6366f1, #9333ea);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.3);
    color: white;
}

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.18);
    padding: 16px;
    border-radius: 15px;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

div[data-testid="stMetricValue"] {
    color: #67e8f9;
}

[data-testid="stSidebar"] {
    background: #0b1120;
    border-right: 1px solid #1e293b;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #111827;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div style="color:#67e8f9;font-weight:700;letter-spacing:2px;">
        ADMISSIONS INTELLIGENCE PLATFORM
    </div>
    <h1>CUET UG <span class="gradient-text">Future Cutoff Predictor</span></h1>
    <p>
        Explore historical admission trends and estimate future cutoffs
        using machine learning and year-wise trend analysis.
    </p>
    <div style="color:#a5b4fc;font-size:13px;margin-top:20px;">
        MACHINE LEARNING &nbsp; • &nbsp; FUTURE FORECASTS
        &nbsp; • &nbsp; DATA-DRIVEN INSIGHTS
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🎓 CUET PREDICTOR")
    st.caption("Future admission insights")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("""
    - 📊 Overview
    - 🔮 Future predictions
    - 🧠 Model performance
    - 🔐 Data and security
    """)

    st.divider()
    st.markdown("### Forecast horizon")
    st.write("2027 – 2030")

    st.caption(
        "Forecasts are estimates, not official admission cutoffs."
    )

# =========================================================
# LOAD BUILT-IN DATASET
# =========================================================
@st.cache_data
def load_data(path_string, modified_time):
    data = pd.read_csv(path_string)
    data.columns = data.columns.astype(str).str.strip()
    return data

if not DATA_PATH.exists():
    st.error("Historical cutoff dataset is not configured yet.")
    st.markdown("""
    **Required setup**

    Create a folder named `data` in your GitHub repository and add:

    `data/cuet_cutoffs.csv`

    The CSV must contain real historical records with these columns:

    - `university`
    - `programme`
    - `year`
    - `category`
    - `round`
    - `cutoff`

    Additional columns, such as `seat_type` or `college`, are allowed.

    Do not use example values as if they were official data.
    """)
    st.stop()

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception:
    st.error(
        "The built-in dataset could not be read. Check the CSV format "
        "and the repository's data folder."
    )
    st.stop()

required = [
    "university", "programme", "year",
    "category", "round", "cutoff"
]

missing = [col for col in required if col not in df.columns]

if missing:
    st.error(
        "The historical dataset is missing required columns: "
        + ", ".join(missing)
    )
    st.stop()

df = df.dropna(subset=required).copy()

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["round"] = pd.to_numeric(df["round"], errors="coerce")
df["cutoff"] = pd.to_numeric(df["cutoff"], errors="coerce")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=required)

if df.empty:
    st.error("No valid historical cutoff records were found.")
    st.stop()

for col in ["university", "programme", "category"]:
    df[col] = df[col].astype(str).str.strip()

df = df[
    (df["university"] != "") &
    (df["programme"] != "") &
    (df["category"] != "")
]

if len(df) < 5:
    st.error(
        "At least five valid historical records are required for this "
        "prototype. More years and records are strongly recommended."
    )
    st.stop()

latest_year = int(df["year"].max())
first_year = int(df["year"].min())

if latest_year >= 2027:
    st.warning(
        "Your dataset contains records from 2027 or later. "
        "Check that these are real observed cutoffs, not forecasts "
        "previously generated by this app."
    )

# =========================================================
# DATA OVERVIEW
# =========================================================
st.markdown('<div class="section-title">📊 Dataset Overview</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Historical Records", f"{len(df):,}")

with m2:
    st.metric("Universities", df["university"].nunique())

with m3:
    st.metric("Programmes", df["programme"].nunique())

with m4:
    st.metric("Latest Data Year", latest_year)

st.caption(
    f"Historical data available from {first_year} to {latest_year}. "
    "Model quality depends on data coverage and reliability."
)

# =========================================================
# MODEL FEATURES
# =========================================================
# Predictors that are known before the cutoff is observed.
# Remove identifiers and the target from the feature set.
excluded = {"cutoff"}

optional_categorical = [
    col for col in [
        "university", "programme", "category",
        "round", "seat_type", "college", "variant",
        "figure_kind", "basis"
    ]
    if col in df.columns and col not in excluded
]

feature_columns = optional_categorical + ["year"]
feature_columns = list(dict.fromkeys(feature_columns))

model_df = df.dropna(subset=["cutoff", "year"]).copy()
model_df = model_df[np.isfinite(model_df["cutoff"])]

categorical_features = [
    col for col in feature_columns if col != "year"
]
numeric_features = ["year"]

# =========================================================
# TIME-BASED MODEL EVALUATION
# =========================================================
@st.cache_resource
def train_and_evaluate(data, features, categorical, numeric):
    usable = data.dropna(subset=["cutoff", "year"]).copy()

    years = sorted(usable["year"].unique())

    result = {
        "pipeline": None,
        "mae": None,
        "r2": None,
        "test_year": None,
        "train_rows": 0,
        "test_rows": 0,
        "evaluation_note": ""
    }

    if len(years) >= 2:
        test_year = years[-1]
        train_data = usable[usable["year"] < test_year]
        test_data = usable[usable["year"] == test_year]

        if len(train_data) >= 5 and len(test_data) >= 1:
            transformers = []

            if categorical:
                transformers.append((
                    "cat",
                    Pipeline([
                        ("imputer", SimpleImputer(
                            strategy="most_frequent"
                        )),
                        ("encoder", OneHotEncoder(
                            handle_unknown="ignore"
                        ))
                    ]),
                    categorical
                ))

            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(
                        strategy="median"
                    ))
                ]),
                numeric
            ))

            preprocessor = ColumnTransformer(transformers)

            evaluation_pipeline = Pipeline([
                ("prep", preprocessor),
                ("model", GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=42,
                    loss="huber"
                ))
            ])

            evaluation_pipeline.fit(
                train_data[features],
                train_data["cutoff"]
            )

            test_predictions = evaluation_pipeline.predict(
                test_data[features]
            )

            result["mae"] = float(mean_absolute_error(
                test_data["cutoff"], test_predictions
            ))

            if (
                len(test_data) >= 2
                and test_data["cutoff"].nunique() > 1
            ):
                result["r2"] = float(r2_score(
                    test_data["cutoff"], test_predictions
                ))

            result["test_year"] = int(test_year)
            result["train_rows"] = int(len(train_data))
            result["test_rows"] = int(len(test_data))
            result["evaluation_note"] = (
                f"Trained on years before {int(test_year)} and "
                f"tested on {int(test_year)}."
            )
        else:
            result["evaluation_note"] = (
                "Not enough earlier records to run a reliable "
                "time-based holdout test."
            )
    else:
        result["evaluation_note"] = (
            "Only one historical year is available. "
            "Time-based model evaluation is not possible."
        )

    # Final model trained on all available historical records
    transformers = []

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical
        ))

    transformers.append((
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric
    ))

    final_pipeline = Pipeline([
        ("prep", ColumnTransformer(transformers)),
        ("model", GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
            loss="huber"
        ))
    ])

    final_pipeline.fit(
        usable[features],
        usable["cutoff"]
    )

    result["pipeline"] = final_pipeline
    return result

with st.spinner("Evaluating historical trends and training the model..."):
    model_result = train_and_evaluate(
        model_df,
        feature_columns,
        categorical_features,
        numeric_features
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.markdown('<div class="section-title">🧠 Model Performance</div>',
            unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)

with p1:
    if model_result["mae"] is not None:
        st.metric("Holdout MAE", f'{model_result["mae"]:.3f}')
    else:
        st.metric("Holdout MAE", "N/A")

with p2:
    if model_result["r2"] is not None:
        st.metric("Holdout R²", f'{model_result["r2"]:.3f}')
    else:
        st.metric("Holdout R²", "N/A")

with p3:
    st.metric(
        "Test Year",
        str(model_result["test_year"])
        if model_result["test_year"] is not None
        else "N/A"
    )

st.info(model_result["evaluation_note"])

st.caption(
    "MAE is the average absolute prediction error, measured in the "
    "same units as the cutoff. R² is a goodness-of-fit metric, not "
    "a percentage accuracy score. A missing or negative R² must not "
    "be presented as high accuracy."
)

# =========================================================
# PREDICTION CONTROLS
# =========================================================
st.markdown('<div class="section-title">🔮 Predict Future Cutoffs</div>',
            unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    universities = sorted(df["university"].unique().tolist())
    selected_university = st.selectbox(
        "Select University",
        universities
    )

university_df = df[df["university"] == selected_university]

with right:
    programmes = sorted(university_df["programme"].unique().tolist())
    selected_programme = st.selectbox(
        "Select Programme",
        programmes
    )

subset = university_df[
    university_df["programme"] == selected_programme
]

c1, c2 = st.columns(2)

with c1:
    categories = sorted(subset["category"].unique().tolist())
    selected_category = st.selectbox(
        "Admission Category",
        categories
    )

filtered = subset[subset["category"] == selected_category]

with c2:
    if "round" in filtered.columns:
        rounds = sorted(
            filtered["round"].dropna().unique().tolist()
        )
        selected_round = st.selectbox(
            "Admission Round",
            rounds
        )
    else:
        selected_round = None

selected_records = filtered.copy()

if selected_round is not None:
    selected_records = selected_records[
        selected_records["round"] == selected_round
    ]

if "seat_type" in selected_records.columns:
    seat_types = sorted(
        selected_records["seat_type"].dropna().astype(str).unique()
    )
    if len(seat_types) > 1:
        selected_seat_type = st.selectbox(
            "Seat Type",
            seat_types
        )
        selected_records = selected_records[
            selected_records["seat_type"].astype(str)
            == selected_seat_type
        ]

st.caption(
    "Select the same admission category and round used in the "
    "historical records for the most meaningful comparison."
)

# =========================================================
# FUTURE FORECAST
# =========================================================
if st.button("✨ Generate 2027–2030 Forecast", type="primary",
             use_container_width=True):

    if selected_records.empty:
        st.error("No historical records match your selected options.")
        st.stop()

    selected_records = selected_records.sort_values("year")

    years_present = sorted(
        selected_records["year"].dropna().astype(int).unique()
    )

    if len(years_present) < 2:
        st.warning(
            "This selection has data for fewer than two distinct years. "
            "A year-to-year trend cannot be estimated reliably. "
            "Add more historical data for this university, programme, "
            "category, and round before relying on future forecasts."
        )

    # For future years, use a fitted linear trend for the exact selected
    # group when at least two distinct years are available.
    # Tree models do not extrapolate year trends well beyond training data.
    forecast_rows = []

    if len(years_present) >= 2:
        trend_model = LinearRegression()
        trend_model.fit(
            selected_records[["year"]],
            selected_records["cutoff"]
        )

        last_observed_year = int(selected_records["year"].max())
        last_observed_cutoff = float(
            selected_records.loc[
                selected_records["year"] == last_observed_year,
                "cutoff"
            ].median()
        )

        trend_slope = float(trend_model.coef_[0])

        for future_year in FUTURE_YEARS:
            estimated = float(
                trend_model.predict(
                    pd.DataFrame({"year": [future_year]})
                )[0]
            )

            forecast_rows.append({
                "Year": future_year,
                "Predicted Cutoff": estimated,
                "Type": "Trend estimate"
            })

        forecast_df = pd.DataFrame(forecast_rows)

        st.session_state["forecast_df"] = forecast_df
        st.session_state["forecast_group"] = (
            selected_university,
            selected_programme,
            selected_category
        )
        st.session_state["last_cutoff"] = last_observed_cutoff
        st.session_state["last_year"] = last_observed_year
        st.session_state["trend_slope"] = trend_slope
        st.session_state["forecast_ready"] = True

    else:
        st.session_state["forecast_ready"] = False
        st.error(
            "No forecast was generated because there are not enough "
            "distinct historical years to estimate a trend. The app "
            "will not fabricate future cutoffs."
        )

# =========================================================
# DISPLAY FORECAST
# =========================================================
if st.session_state.get("forecast_ready", False):
    forecast_df = st.session_state["forecast_df"]

    st.markdown(
        '<div class="section-title">📈 Your Forecast Results</div>',
        unsafe_allow_html=True
    )

    k1, k2, k3 = st.columns(3)

    with k1:
        st.metric(
            "Latest Historical Cutoff",
            f'{st.session_state["last_cutoff"]:,.2f}',
            help=f'Observed in {st.session_state["last_year"]}'
        )

    with k2:
        st.metric(
            "2030 Estimated Cutoff",
            f'{forecast_df.iloc[-1]["Predicted Cutoff"]:,.2f}'
        )

    with k3:
        direction = (
            "Increasing"
            if st.session_state["trend_slope"] > 0
            else "Decreasing"
            if st.session_state["trend_slope"] < 0
            else "Stable"
        )
        st.metric("Historical Trend", direction)

    chart_data = forecast_df.copy()
    chart_data["Year"] = chart_data["Year"].astype(str)

    fig = px.line(
        chart_data,
        x="Year",
        y="Predicted Cutoff",
        markers=True,
        title="Estimated Cutoff Trend (2027–2030)",
        template="plotly_dark"
    )

    fig.update_traces(
        line=dict(width=4, color="#67e8f9"),
        marker=dict(size=10, color="#a78bfa")
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title_font_size=20,
        xaxis_title="Admission Year",
        yaxis_title="Estimated Cutoff",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        forecast_df.style.format({
            "Predicted Cutoff": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="cuet_future_cutoff_estimates.csv",
        mime="text/csv"
    )

    st.warning(
        "These are statistical trend estimates, not official cutoffs. "
        "They assume the historical trend continues. Changes in exam "
        "difficulty, applicant numbers, seat availability, policies, "
        "and counselling rounds can substantially change future cutoffs."
    )

# =========================================================
# HISTORICAL DATA VISUALIZATION
# =========================================================
st.markdown(
    '<div class="section-title">📚 Historical Cutoff Trends</div>',
    unsafe_allow_html=True
)

history = selected_records.sort_values("year").copy()

if not history.empty:
    hist_fig = px.line(
        history,
        x="year",
        y="cutoff",
        markers=True,
        title="Observed Historical Cutoffs",
        template="plotly_dark"
    )

    hist_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis_title="Year",
        yaxis_title="Observed Cutoff"
    )

    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.info("No historical data is available for this selection.")

# =========================================================
# SECURITY / TRANSPARENCY
# =========================================================
st.markdown(
    '<div class="section-title">🔐 Data & Security</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="glass-card">
    <div style="font-size:18px;font-weight:700;color:#67e8f9;">
        Privacy-conscious design
    </div>
    <ul style="color:#cbd5e1;line-height:1.9;">
        <li>No user dataset upload control.</li>
        <li>Historical data is read from the project repository.</li>
        <li>No passwords or API keys are required by this application.</li>
        <li>Predictions use validated numeric cutoff and year values.</li>
        <li>Model evaluation results are shown transparently.</li>
    </ul>
    <div style="color:#fbbf24;font-size:13px;">
        Note: Public repository data and app code are not secret.
        Add authentication and rate limiting separately if required.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div style="text-align:center;color:#94a3b8;padding:15px;">
    <b>CUET UG Future Cutoff Predictor</b><br>
    Machine Learning • Trend Analysis • Admission Insights<br>
    <span style="font-size:12px;">
        For educational and planning purposes only.
        Always verify official university admission notices.
    </span>
</div>
""", unsafe_allow_html=True)
