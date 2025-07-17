import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import lime
from lime.lime_text import LimeTextExplainer

st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection App (BERT + LIME + Streamlit)")

# Upload dataset
uploaded_fake = st.file_uploader("Upload Fake.csv file", type="csv")
uploaded_true = st.file_uploader("Upload True.csv file", type="csv")

if uploaded_fake and uploaded_true:
    fake = pd.read_csv(uploaded_fake)
    true = pd.read_csv(uploaded_true)
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
    df.dropna(subset=["text"], inplace=True)
    st.success("Datasets loaded successfully!")

    # Train model
    with st.spinner("Generating BERT embeddings and training model..."):
        sample_df = df.sample(2000, random_state=42)
        texts = sample_df["text"].tolist()
        labels = sample_df["label"].values
        bert = SentenceTransformer("distilbert-base-nli-mean-tokens")
        X = bert.encode(texts)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, labels)

    # Prediction section
    st.subheader("🔎 Test the Model")
    user_input = st.text_area("Paste a news article here:", height=300)

    if st.button("Analyze"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing with BERT..."):
                emb = bert.encode([user_input])
                prediction = clf.predict(emb)[0]
                proba = clf.predict_proba(emb)[0][prediction]
            label_name = "REAL" if prediction == 1 else "FAKE"
            st.success(f"Prediction: **{label_name}** with confidence {proba:.2f}")

            if "you won't believe" in user_input.lower() or "shocking" in user_input.lower():
                st.info("💡 This article might be fake due to emotionally charged phrases like 'shocking' or 'you won’t believe'.")

            # LIME explanation
            st.subheader("📌 LIME Explanation")
            explainer = LimeTextExplainer(class_names=["Fake", "Real"])
            predict_fn = lambda x: clf.predict_proba(bert.encode(x))
            exp = explainer.explain_instance(user_input, predict_fn, num_features=10)
            st.components.v1.html(exp.as_html(), height=600, scrolling=True)
else:
    st.warning("Please upload both Fake.csv and True.csv files to continue.")
