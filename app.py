import streamlit as st
import numpy as np
from model import FEATURES, predict_sample, get_top_features

st.set_page_config(page_title="Breast Cancer SVM Classifier", page_icon="🧪")

st.title("🧪 Breast Cancer Risk Classifier (SVM)")
st.subheader("🔬 Predict whether a tumor is malignant or benign")
st.caption("⚠ For learning only — not a medical tool")

st.markdown("---")

# Ask user to input features
inputs = []
cols = st.columns(3)




for i, feature in enumerate(FEATURES):
    col = cols[i % 3]
    val = col.number_input(
        feature,
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=0.1,
        format="%.3f"
    )
    inputs.append(val)

if st.button("🔍 Predict"):
    result, confidence = predict_sample(inputs)

    st.success(f"Prediction: **{result.upper()}**")
    st.metric("Confidence", f"{confidence}%")

    st.markdown("---")
    st.write("🔥 **Top influential features**")
    st.table({
        "Feature": [x[0] for x in get_top_features()],
        "Weight": [x[1] for x in get_top_features()],
    })
else:
    st.info("Fill the fields → Click **Predict**")
