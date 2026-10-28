"""Streamlit app: classify an SMS message as spam or ham.

Run:  streamlit run app.py
"""
from pathlib import Path

import joblib
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️")
st.title("SMS Spam Classifier")
st.caption("TF-IDF + calibrated Linear SVM trained on the UCI SMS Spam Collection.")

model = load_model()
text = st.text_area("Paste an SMS message", height=120,
                    placeholder="WINNER!! You have won a free prize. Call 09061701461 now to claim.")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Enter a message first.")
    else:
        p = float(model.predict_proba([text])[0, 1])
        st.metric("Spam probability", f"{p:.1%}")
        if p >= 0.5:
            st.error("Likely SPAM")
        else:
            st.success("Likely legitimate (ham)")
