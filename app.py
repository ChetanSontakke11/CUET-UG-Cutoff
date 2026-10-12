import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="CUET UG | Cutoff Intelligence", page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "cuet_cutoffs.csv"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 8% 0%,rgba(30,64,175,.23),transparent 34%),radial-gradient(ellipse at 95% 18%,rgba(124,58,237,.15),transparent 28%),#080d18;color:#edf2ff;font-family:'DM Sans',sans-serif}
[data-testid="stHeader"]{background:rgba(8,13,24,.82)} .block-container{max-width:1380px;padding-top:1.6rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#101a2c!important;border-right:1px solid #344563!important}
[data-testid="stSidebar"] > div{background:#101a2c!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small{color:#e2e8f0!important}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:#f8fafc!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#a8b8d1!important}
[data-testid="stSidebar"] hr{border-color:#344563!important}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a{color:#7dd3fc!important}
[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1.25rem}
[data-testid="stSidebar"] [data-testid="stButton"] button{background:#172943!important;border:1px solid #344563!important;color:#f8fafc!important}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important}
.hero{padding:34px 36px;border-radius:26px;background:linear-gradient(115deg,#101c37,#171c42,#182c4b);border:1px solid #334155;box-shadow:0 20px 70px #0005;animation:appear .65s ease-out}
.eyebrow{color:#7dd3fc;font-size:11px;font-weight:700;letter-spacing:2.4px}
.hero h1{color:white;font-size:clamp(32px,4vw,50px);line-height:1.07;margin:14px 0 12px}
.accent{background:linear-gradient(90deg,#67e8f9,#a5b4fc,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:#b8c5dc;font-size:15px;max-width:720px;line-height:1.7}
.pill{display:inline-block;margin-top:12px;padding:6px 10px;border-radius:99px;border:1px solid #7dd3fc44;background:#0ea5e914;color:#bae6fd;font-size:11px}
.metric-card,.panel{background:linear-gradient(145deg,#121d31f2,#0d1423f5);border:1px solid #273650;border-radius:16px;padding:18px;animation:appear .45s ease-out}
.metric-card{min-height:105px;transition:transform .2s,border-color .2s}.metric-card:hover{transform:translateY(-3px);border-color:#5272a8}
.metric-label{color:#9aabc5;font-size:11px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase}
.metric-value{color:#eaf4ff;font-size:26px;font-weight:700;margin-top:7px}.metric-note{color:#7186a6;font-size:11px;margin-top:4px}
.side-nav{display:grid;gap:9px;margin:12px 0 18px}.side-nav div{display:flex;align-items:center;gap:10px;padding:10px 11px;border-radius:9px;background:#172943;border:1px solid #2b405f;color:#e2e8f0;font-size:12px;font-weight:600}.side-nav span{color:#7dd3fc;font-size:10px;font-weight:800;letter-spacing:1px}\n.section-kicker{color:#7dd3fc;font-size:11px;font-weight:700;letter-spacing:1.7px;text-transform:uppercase;margin-top:24px}
.section-heading{font-size:24px;font-weight:700;color:#f1f5f9;margin:3px 0 8px}.muted{color:#9aabc5;font-size:13px;line-height:1.65}
div.stButton>button,div.stDownloadButton>button{border-radius:11px;border:1px solid #3b82f6;min-height:44px;font-weight:700;background:linear-gradient(100deg,#0e7490,#4f46e5);color:white;transition:.2s}
div.stButton>button:hover,div.stDownloadButton>button:hover{border-color:#7dd3fc;transform:translateY(-2px);color:white;box-shadow:0 8px 24px #3b82f62e}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#101a2c!important;border-color:#2c3b55!important;border-radius:10px!important}
div[data-testid="stMetric"]{background:#101a2c;border:1px solid #273650;padding:15px;border-radius:14px}
div[data-testid="stMetricLabel"]{color:#a8b8d1} div[data-testid="stMetricValue"]{color:#7dd3fc}
@keyframes appear{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
footer,#MainMenu{visibility:hidden}
</style>
""", unsafe_allow_html=True)

def demo_data():
    rows = []
    sample = {
        "Demo Central University": {
            "B.Sc. Computer Science": {"General":[188,184,180,177,173],"OBC":[178,174,171,167,164],"SC":[159,155,151,148,144]},
            "B.A. Economics": {"General":[181,178,175,171,168],"OBC":[171,168,164,161,158],"SC":[153,149,146,142,139]}},
        "Demo National University": {
            "B.Com": {"General":[192,189,185,182,179],"OBC":[182,179,175,172,169],"SC":[163,160,156,153,150]},
            "B.A. Political Science": {"General":[176,173,169,166,163],"OBC":[166,163,159,156,153],"SC":[148,145,141,138,135]}},
        "Demo State University": {
            "B.Sc. Mathematics": {"General":[172,169,166,162,159],"OBC":[162,159,156,152,149],"SC":[145,142,139,135,132]},
            "B.A. English": {"General":[168,165,162,158,155],"OBC":[158,155,152,148,145],"SC":[141,138,135,131,128]}}
    }
    for uni, programmes in sample.items():
        for prog, categories in programmes.items():
            for cat, vals in categories.items():
                for year, val in zip(range(2022, 2027), vals):
                    for rnd, delta in [(1,0),(2,-7)]:
                        rows.append({"university":uni,"programme":prog,"year":year,"category":cat,"round":rnd,"cutoff":val+delta})
    return pd.DataFrame(rows)

# A local real-data CSV overrides the explicitly fictional demo data.
if DATA_PATH.exists():
    try:
        df = pd.read_csv(DATA_PATH)
        demo_mode = False
    except Exception:
        st.error("Could not read data/cuet_cutoffs.csv. Check that it is a valid CSV.")
        st.stop()
else:
    df = demo_data()
    demo_mode = True

df.columns = df.columns.astype(str).str.strip().str.lower()
required = ["university","programme","year","category","round","cutoff"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Dataset missing required columns: " + ", ".join(missing))
    st.stop()
for col in ["year","round","cutoff"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.replace([np.inf,-np.inf],np.nan).dropna(subset=required).copy()
for col in ["university","programme","category"]:
    df[col] = df[col].astype(str).str.strip()
df = df[(df["university"]!="")&(df["programme"]!="")&(df["category"]!="")]
if df.empty or len(df)<5:
    st.error("Not enough valid records. Check the dataset.")
    st.stop()

st.markdown("""
<div class="hero">
 <div class="eyebrow">ADMISSIONS INTELLIGENCE · MACHINE LEARNING</div>
 <h1>CUET UG<br><span class="accent">Cutoff Predictor</span></h1>
 <p>Explore historical cutoff patterns and estimate future admission cutoffs with a clear, data-driven dashboard built for planning your next move.</p>
 <span class="pill">FORECAST WINDOW · CHOOSE 1–8 FUTURE YEARS</span>
</div>
""", unsafe_allow_html=True)

if demo_mode:
    st.warning("DEMO MODE: These are fictional sample records to demonstrate the app, NOT official CUET cutoffs. Add verified data/cuet_cutoffs.csv in GitHub to use real records.")
else:
    st.success("Historical dataset loaded from the project repository.")

with st.sidebar:
    st.markdown("## 🎓 CUET / INSIGHTS")
    st.caption("Future cutoff dashboard")
    st.divider()
    st.markdown("**Dashboard sections**")
    st.markdown("01 · Overview\n\n02 · Forecast explorer\n\n03 · Model evaluation\n\n04 · Data transparency")
    st.divider()
    st.markdown("**Forecast horizon**")
    years_to_forecast = st.selectbox("How many future years?", options=list(range(1, 9)), index=3, format_func=lambda n: f"{n} year" if n == 1 else f"{n} years")
    first_forecast_year = max(int(df["year"].max()) + 1, date.today().year + 1)
    forecast_years = list(range(first_forecast_year, first_forecast_year + years_to_forecast))
    st.caption(f"Forecasting {forecast_years[0]}–{forecast_years[-1]}")
    st.caption("Estimates only. Verify official admission notices.")

def metric_card(label,value,note):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)

first_year=int(df["year"].min()); latest_year=int(df["year"].max())
st.markdown('<div class="section-kicker">OVERVIEW</div><div class="section-heading">Your data at a glance</div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
with a: metric_card("Historical records",f"{len(df):,}","Rows loaded")
with b: metric_card("Universities",f"{df.university.nunique():,}","Institutions")
with c: metric_card("Programmes",f"{df.programme.nunique():,}","Courses")
with d: metric_card("Data coverage",f"{first_year}–{latest_year}","Available years")

st.markdown('<div class="section-kicker">01 / FORECAST EXPLORER</div><div class="section-heading">Find your future cutoff</div><div class="muted">Choose a university, course, category and round. Estimates use that selection’s historical trend.</div>',unsafe_allow_html=True)
col1,col2=st.columns(2)
with col1:
    university=st.selectbox("University",sorted(df.university.unique().tolist()))
uni_df=df[df.university==university]
with col2:
    programme=st.selectbox("Programme / Course",sorted(uni_df.programme.unique().tolist()))
prog_df=uni_df[uni_df.programme==programme]
col3,col4=st.columns(2)
with col3:
    category=st.selectbox("Admission category",sorted(prog_df.category.unique().tolist()))
cat_df=prog_df[prog_df.category==category]
with col4:
    round_value=st.selectbox("Counselling round",sorted(cat_df["round"].unique().tolist()))
selected=cat_df[cat_df["round"]==round_value].sort_values("year").copy()

# Honest time-based holdout evaluation of the global ML model.
def evaluate(data):
    years=sorted(data.year.unique())
    if len(years)<2:
        return None,None,None,"Not enough distinct years for time-based evaluation."
    test_year=years[-1]
    train=data[data.year<test_year].copy()
    test=data[data.year==test_year].copy()
    cats=["university","programme","category","round"]
    nums=["year"]
    if len(train)<10 or len(test)<2:
        return None,None,int(test_year),"Not enough records for a useful holdout test."
    prep=ColumnTransformer([
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),cats),
        ("num",Pipeline([("impute",SimpleImputer(strategy="median"))]),nums)
    ])
    pipe=Pipeline([("prep",prep),("model",GradientBoostingRegressor(n_estimators=100,learning_rate=.05,max_depth=2,random_state=42))])
    pipe.fit(train[cats+nums],train.cutoff)
    pred=pipe.predict(test[cats+nums])
    mae=float(mean_absolute_error(test.cutoff,pred))
    r2=float(r2_score(test.cutoff,pred)) if test.cutoff.nunique()>1 else None
    return mae,r2,int(test_year),f"Trained on years before {int(test_year)}; tested on {int(test_year)}."

mae,r2,test_year,eval_note=evaluate(df)
st.markdown('<div class="section-kicker">02 / MODEL HEALTH</div><div class="section-heading">Performance, without fake accuracy</div>',unsafe_allow_html=True)
e1,e2,e3=st.columns(3)
with e1: st.metric("Holdout MAE",f"{mae:.2f}" if mae is not None else "N/A")
with e2: st.metric("Holdout R²",f"{r2:.3f}" if r2 is not None else "N/A")
with e3: st.metric("Test year",str(test_year) if test_year is not None else "N/A")
st.caption(eval_note+" MAE is measured in cutoff units; R² is not a percentage accuracy score.")

st.markdown('<div class="section-kicker">03 / FUTURE OUTLOOK</div><div class="section-heading">Your future cutoff estimate</div>',unsafe_allow_html=True)
if selected.year.nunique()<2:
    st.info("This selection needs at least two distinct historical years before a trend can be estimated. Add more verified records or choose another combination.")
else:
    annual=selected.groupby("year",as_index=False).cutoff.median().sort_values("year")
    trend=LinearRegression().fit(annual[["year"]],annual["cutoff"])
    # Keep the prediction column name identical to the feature used during model fitting.
    forecast=pd.DataFrame({"year":forecast_years})
    forecast["Estimated cutoff"]=trend.predict(forecast[["year"]])
    forecast=forecast.rename(columns={"year":"Year"})
    last_year=int(annual.year.max())
    last_cutoff=float(annual.loc[annual.year==last_year,"cutoff"].iloc[0])
    slope=float(trend.coef_[0])
    k1,k2,k3=st.columns(3)
    with k1: st.metric("Latest observed cutoff",f"{last_cutoff:,.2f}",f"Year {last_year}")
    with k2: st.metric("2030 trend estimate",f'{forecast.iloc[-1]["Estimated cutoff"]:,.2f}')
    with k3: st.metric("Observed direction","Rising" if slope>.05 else "Falling" if slope<-.05 else "Mostly stable")
    st.markdown('<div class="panel"><b>Cutoff trajectory</b><div class="muted">Observed values and future trend estimates</div></div>',unsafe_allow_html=True)
    history=annual.rename(columns={"year":"Year","cutoff":"Cutoff"}).set_index("Year")
    future=forecast.rename(columns={"Estimated cutoff":"Cutoff"}).set_index("Year")
    combined=pd.concat([history.assign(Type="Historical")["Cutoff"],future.assign(Type="Estimate")["Cutoff"]],axis=1)
    combined.columns=["Historical cutoff","Estimated cutoff"]
    st.line_chart(combined,use_container_width=True)
    st.dataframe(forecast.style.format({"Estimated cutoff":"{:,.2f}"}),use_container_width=True,hide_index=True)
    st.download_button("Download forecast CSV",forecast.to_csv(index=False).encode("utf-8"),"cuet_cutoff_forecast.csv","text/csv")
    st.warning("These are trend extrapolations, not official cutoffs or guaranteed admission outcomes. Future cutoffs can change with applicant numbers, exam difficulty, seats and admission rules.")

st.markdown('<div class="section-kicker">04 / HISTORICAL TRENDS</div><div class="section-heading">Past cutoff movement</div>',unsafe_allow_html=True)
if not selected.empty:
    hist=selected.groupby("year").cutoff.median().sort_index()
    st.line_chart(hist,use_container_width=True)
else:
    st.info("No matching historical records.")

st.markdown('<div class="section-kicker">05 / TRANSPARENCY</div><div class="section-heading">Data & security</div>',unsafe_allow_html=True)
s1,s2=st.columns(2)
with s1:
    st.markdown('<div class="panel"><b style="color:#7dd3fc">Designed for safer use</b><p class="muted">No user upload control. Data is read from the repository. No passwords or API keys are required. Evaluation results are displayed transparently.</p></div>',unsafe_allow_html=True)
with s2:
    st.markdown('<div class="panel"><b style="color:#c4b5fd">Know the limits</b><p class="muted">Public repository data and source code are not secret. This app does not guarantee admission. Replace demo records with verified official data before relying on estimates.</p></div>',unsafe_allow_html=True)

st.divider()
st.markdown('<div style="text-align:center;color:#8192ad;font-size:12px;line-height:1.9"><b style="color:#cbd5e1">CUET UG · CUTOFF INTELLIGENCE</b><br>Machine learning · Historical analysis · Future estimates<br>For educational planning only · Verify official admission notices</div>',unsafe_allow_html=True)
