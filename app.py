import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ── Page config ──
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🔍",
    layout="wide"
)

# ── Load model ──
MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Title ──
st.title("Credit Card Fraud Detection")
st.markdown("Enter transaction details below to check if it is fraudulent.")
st.divider()

# ── Sidebar inputs ──
st.sidebar.header("Transaction inputs")
st.sidebar.markdown("Adjust the feature values:")

# Input sliders for key features
amount = st.sidebar.number_input("Transaction Amount (€)", min_value=0.0,
                                  max_value=25000.0, value=100.0, step=0.01)

v_features = {}
important_features = ['V1', 'V2', 'V3', 'V4', 'V10', 'V12', 'V14', 'V17']

for feat in important_features:
    v_features[feat] = st.sidebar.slider(
        feat, min_value=-10.0, max_value=10.0, value=0.0, step=0.01
    )

# Fill remaining V features with 0 (neutral)
all_features = {}
for i in range(1, 29):
    col = f'V{i}'
    all_features[col] = v_features.get(col, 0.0)
all_features['Amount'] = amount

# ── Predict button ──
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Run prediction")
    if st.button("Check transaction", type="primary", use_container_width=True):

        input_df = pd.DataFrame([all_features])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        st.divider()

        if prediction == 1:
            st.error("FRAUD DETECTED")
            st.metric("Fraud probability", f"{probability[1]*100:.1f}%")
        else:
            st.success("LEGITIMATE TRANSACTION")
            st.metric("Fraud probability", f"{probability[1]*100:.1f}%")

        # ── SHAP explanation ──
        st.subheader("Why did the model decide this?")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=float(explainer.expected_value),
                data=input_df.iloc[0].values,
                feature_names=input_df.columns.tolist()
            )
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

with col2:
    st.subheader("Current input values")
    display_df = pd.DataFrame([all_features]).T
    display_df.columns = ['Value']
    st.dataframe(display_df, use_container_width=True, height=400)

# ── Footer ──
st.divider()
st.caption("Built with XGBoost + SHAP | Credit Card Fraud Detection Project")
