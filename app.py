from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="SMS Spam Checker")

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "spam_model.joblib"
    return joblib.load(model_path)

saved = load_model()
vectorizer = saved["vectorizer"]
model = saved["model"]

st.title("SMS Spam Checker")
st.write("Paste a text message to estimate whether it is spam.")

message = st.text_area("Your message")

threshold = st.slider(
    "Spam decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

if st.button("Check message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        # Use the saved converter. Do not fit it again.
        features = vectorizer.transform([message])

        # Find the probability column for class 1 = spam.
        spam_column = list(model.classes_).index(1)
        probability = model.predict_proba(features)[0, spam_column]

        st.metric("Estimated spam probability", f"{probability:.1%}")

        if probability >= threshold:
            st.error("Prediction: Spam")
        else:
            st.success("Prediction: Not spam")

        st.caption(
            f"Messages with a spam probability of "
            f"{threshold:.0%} or higher are labelled Spam. "
            "Predictions can be wrong."
        )
