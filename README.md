# CUET UG Cutoff Predictor

A Streamlit UI for running the saved Gradient Boosting regression model against CUET UG cutoff features.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, usually http://localhost:8501.

## Model inputs and limitations

The app expects a saved `gradient_boosting_model.pkl` alongside `app.py`. The assumed feature order is:

`university_id`, `university`, `year`, `programme_id`, `programme`, `college`, `variant`, `round`, `category`, `seat_type`, `figure_kind`, `basis`, `source_id`, `source_page`.

The model file does not contain the original training dataset, target description, or category mappings. Use the exact preprocessing and numeric encodings from training. The displayed result is an estimate, not an official cutoff or admission guarantee.

Only load pickle files you trust; pickle deserialization can execute code.
