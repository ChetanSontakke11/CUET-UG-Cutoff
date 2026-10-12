```python
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="CUET UG Cutoff Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 CUET UG Cutoff Prediction System")
st.write(
    "Upload historical cutoff data, train a machine-learning model, "
    "and estimate a cutoff for your selected admission details."
)

st.info(
    "Predictions are estimates based on the uploaded data. "
    "They are not official admission cutoffs or guarantees."
)

# -----------------------------
# 1. Upload dataset
# -----------------------------
st.header("1. Upload Historical Cutoff Dataset")

uploaded_file = st.file_uploader(
    "Upload your CUET cutoff CSV file",
    type=["csv"]
)

if uploaded_file is None:
    st.markdown("""
    **Your CSV should contain columns such as:**
    - University
    - Programme
    - Year
    - Category
    - Round
    - Cutoff score or closing rank
    """)
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

# Clean column names
df.columns = df.columns.astype(str).str.strip()

# Remove completely empty rows and columns
df = df.dropna(how="all").dropna(axis=1, how="all")

if df.empty:
    st.error("The uploaded CSV contains no usable data.")
    st.stop()

st.success(f"Dataset loaded: {len(df)} rows and {len(df.columns)} columns")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# 2. Select prediction target
# -----------------------------
st.header("2. Select the Cutoff Column")

target_options = [
    col for col in df.columns
    if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0
]

if not target_options:
    st.error(
        "No numeric columns were found. Your dataset must contain "
        "a numeric cutoff score or closing rank."
    )
    st.stop()

default_target = next(
    (
        col for col in target_options
        if any(word in col.lower()
               for word in ["cutoff", "closing_rank", "rank", "score"])
    ),
    target_options[-1]
)

target = st.selectbox(
    "Which column should the model predict?",
    target_options,
    index=target_options.index(default_target)
)

st.caption(
    "Choose the actual cutoff value you want to predict. "
    "Do not choose university IDs or programme IDs as the target."
)

# -----------------------------
# 3. Prepare training data
# -----------------------------
data = df.copy()

# Convert target to numeric
data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])

# Remove rows with infinite target values
data = data[np.isfinite(data[target])]

feature_columns = [col for col in data.columns if col != target]

if not feature_columns:
    st.error("The dataset needs at least one input feature besides the target.")
    st.stop()

# Exclude columns that have no usable values
feature_columns = [
    col for col in feature_columns
    if data[col].notna().sum() > 0
]

if not feature_columns:
    st.error("No usable input columns were found.")
    st.stop()

# Infer numeric and categorical columns
numeric_features = []
categorical_features = []

for col in feature_columns:
    converted = pd.to_numeric(data[col], errors="coerce")
    non_null = data[col].notna().sum()

    if non_null > 0 and converted.notna().sum() / non_null >= 0.9:
        data[col] = converted
        numeric_features.append(col)
    else:
        data[col] = data[col].astype("string").fillna("Unknown")
        categorical_features.append(col)

# Fill missing feature values
for col in numeric_features:
    data[col] = data[col].fillna(data[col].median())

for col in categorical_features:
    data[col] = data[col].fillna("Unknown").astype(str)

X = data[feature_columns]
y = data[target]

if len(data) < 2:
    st.error("At least two valid historical records are needed to train.")
    st.stop()

# -----------------------------
# 4. Train model
# -----------------------------
st.header("3. Train the Prediction Model")

st.write(f"Valid training records: **{len(data)}**")
st.write(f"Input features: **{len(feature_columns)}**")

if st.button("Train Model", type="primary"):
    try:
        transformers = []

        if numeric_features:
            transformers.append(
                ("numeric", "passthrough", numeric_features)
            )

        if categorical_features:
            transformers.append(
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features
                )
            )

        preprocessor = ColumnTransformer(transformers=transformers)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            ))
        ])

        # Evaluate on a holdout set when enough records are available
        if len(data) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            evaluation_model = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.05,
                    max_depth=3,
                    random_state=42
                ))
            ])

            evaluation_model.fit(X_train, y_train)
            predictions = evaluation_model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            st.session_state["mae"] = float(mae)

            if len(y_test) >= 2 and y_test.nunique() > 1:
                st.session_state["r2"] = float(
                    r2_score(y_test, predictions)
                )
            else:
                st.session_state["r2"] = None
        else:
            st.session_state["mae"] = None
            st.session_state["r2"] = None

        # Train final model on all available records
        model.fit(X, y)

        st.session_state["model"] = model
        st.session_state["feature_columns"] = feature_columns
        st.session_state["numeric_features"] = numeric_features
        st.session_state["categorical_features"] = categorical_features
        st.session_state["target"] = target
        st.session_state["training_data"] = data[feature_columns].copy()
        st.session_state["training_rows"] = len(data)

        st.success("Model trained successfully!")

    except Exception as e:
        st.error(f"Model training failed: {e}")

# -----------------------------
# 5. Show evaluation metrics
# -----------------------------
if "model" in st.session_state:
    st.header("4. Model Evaluation")

    mae = st.session_state.get("mae")
    r2 = st.session_state.get("r2")

    col1, col2 = st.columns(2)

    with col1:
        if mae is not None:
            st.metric("Mean Absolute Error (MAE)", f"{mae:.3f}")
        else:
            st.metric("MAE", "Not calculated")
            st.caption("More records are recommended for evaluation.")

    with col2:
        if r2 is not None:
            st.metric("R² Score", f"{r2:.3f}")
        else:
            st.metric("R² Score", "Not calculated")

    st.caption(
        "MAE measures the average absolute prediction error. "
        "R² indicates how well predictions fit the test data; "
        "a negative R² is possible. These metrics do not guarantee "
        "accuracy for future admission years."
    )

    # -----------------------------
    # 6. Enter details and predict
    # -----------------------------
    st.header("5. Predict a Cutoff")

    st.write("Enter the university, programme, year, and other details.")

    feature_columns = st.session_state["feature_columns"]
    numeric_features = st.session_state["numeric_features"]
    categorical_features = st.session_state["categorical_features"]
    training_data = st.session_state["training_data"]

    with st.form("prediction_form"):
        input_values = {}

        for col in feature_columns:
            if col in numeric_features:
                numeric_series = pd.to_numeric(
                    training_data[col], errors="coerce"
                ).dropna()

                if numeric_series.empty:
                    default_value = 0.0
                else:
                    default_value = float(numeric_series.median())

                # Allow future years beyond the historical range
                if "year" in col.lower():
                    input_values[col] = st.number_input(
                        col,
                        min_value=1900,
                        max_value=2200,
                        value=max(2027, int(default_value)),
                        step=1
                    )
                else:
                    input_values[col] = st.number_input(
                        col,
                        value=default_value
                    )

            else:
                options = sorted(
                    training_data[col].astype(str).unique().tolist()
                )

                if not options:
                    options = ["Unknown"]

                # Offer a text field for year-like values only if needed;
                # other categorical features use values from the dataset.
                input_values[col] = st.selectbox(
                    col,
                    options=options
                )

        submitted = st.form_submit_button(
            "Predict Cutoff",
            type="primary"
        )

    if submitted:
        try:
            input_df = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )

            prediction = st.session_state["model"].predict(input_df)[0]

            st.subheader("Predicted Cutoff")
            st.metric(
                label=f"Estimated {st.session_state['target']}",
                value=f"{prediction:,.2f}"
            )

            st.caption(
                "This estimate is generated from the uploaded historical "
                "records. Verify official university admission notices "
                "before making admission decisions."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# -----------------------------
# 7. Footer
# -----------------------------
st.divider()
st.caption("CUET UG Cutoff Predictor | Machine Learning Project")
