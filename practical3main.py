import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------
# Sample Dataset
# -------------------------
data = {
    "Email": [
        "Congratulations! You've won a free iPhone",
        "Claim your lottery prize now",
        "Exclusive deal just for you",
        "Act fast! Limited-time offer",
        "Click here to secure your reward",
        "Hello, how are you today",
        "Please find the attached report",
        "Let's schedule a meeting tomorrow"
    ],
    "Label": [
        "Spam", "Spam", "Spam", "Spam", "Spam",
        "Not Spam", "Not Spam", "Not Spam"
    ]
}

df = pd.DataFrame(data)

# -------------------------
# Title
# -------------------------
st.title("📧 Simple Spam Email Classifier")

st.subheader("📊 Sample Dataset")
st.dataframe(df)

# -------------------------
# Model Training (Auto)
# -------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["Email"])
y = df["Label"]

model = MultinomialNB()
model.fit(X, y)

# -------------------------
# Prediction Section
# -------------------------
st.subheader("🔍 Check Your Email")

user_input = st.text_area("Enter email text:")

if st.button("Check"):
    if user_input.strip() != "":
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)

        if prediction[0] == "Spam":
            st.error("🚫 This Email is SPAM")
        else:
            st.success("✅ This Email is NOT Spam")
    else:
        st.warning("Please enter some text.")
