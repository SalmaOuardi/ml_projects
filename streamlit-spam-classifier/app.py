import streamlit as st
from src.predict import predict_spam

st.set_page_config(page_title="📩 Spam Classifier", layout="centered")

st.title("📩 Spam Message Classifier")
st.write("Enter a message below to check if it's **Spam** or **Ham**.")

# Text input
user_input = st.text_area("💬 Your Message", height=150)

if st.button("🔍 Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        label, confidence = predict_spam(user_input)
        st.success(f"**Prediction:** {label} \n\n**Confidence:** {confidence * 100:.1f}%")

# Optional: Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Salma")
