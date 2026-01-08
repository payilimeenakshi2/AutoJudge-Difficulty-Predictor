# =========================================
# STREAMLIT APP: AutoJudge Difficulty Predictor
# =========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import re
import numpy as np
from scipy.sparse import hstack

# =========================================
# LOAD SAVED MODELS
# =========================================
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")
clf_model = joblib.load("rf_classifier.pkl")
reg_model = joblib.load("rf_regressor.pkl")

# =========================================
# TEXT CLEANING FUNCTION
# =========================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================================
# STREAMLIT UI
# =========================================
st.set_page_config(page_title="AutoJudge Difficulty Predictor", layout="wide")
st.title("📝 AutoJudge: Programming Problem Difficulty Predictor")
st.write("Paste the problem details below to predict its difficulty.")

# -------------------------
# User input
# -------------------------
problem_desc = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")

if st.button("Predict Difficulty"):
    if not problem_desc.strip():
        st.warning("Please enter at least the Problem Description.")
    else:
        combined_text = problem_desc + " " + input_desc + " " + output_desc
        cleaned_text = clean_text(combined_text)

        # TF-IDF features
        X_tfidf = tfidf_vectorizer.transform([cleaned_text])

        # Handcrafted features
        text_length = len(cleaned_text)
        math_symbols = "+-*/=<>^"
        math_symbol_count = sum(cleaned_text.count(s) for s in math_symbols)

        keywords = ["graph", "dp", "dynamic programming", "recursion", "tree", "greedy"]
        keyword_features = [cleaned_text.count(k) for k in keywords]

        X_extra = np.array([[text_length, math_symbol_count] + keyword_features])

        # Combine features
        X_final = hstack([X_tfidf, X_extra])

        # Predictions
        class_pred_enc = clf_model.predict(X_final)[0]
        class_pred = label_encoder.inverse_transform([class_pred_enc])[0]
        score_pred = reg_model.predict(X_final)[0]

        st.success(f"**Predicted Difficulty Class:** {class_pred}")
        st.success(f"**Predicted Difficulty Score:** {score_pred:.2f}")

