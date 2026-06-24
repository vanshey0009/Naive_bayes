import streamlit as st
import pickle
import time

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page title
st.title("📧 SMS Spam Detection System")
st.subheader("Machine Learning Based Text Analytics")

# Instructions
st.info("""
📌 Instructions

• Enter your SMS message below

• Type 'exit' to quit

• Never share sensitive information online
""")

# Input
sms = st.text_area("📩 Enter Message")

# Exit option
if sms.strip().lower() == "exit":
    st.success("👋 Thank you for using SMS Spam Detection System")
    st.stop()

# Predict button
if st.button("🔍 Analyze Message"):

    if not sms.strip():
        st.warning("Please enter a message.")

    elif len(sms.split()) < 2:
        st.warning("⚠ Message too short. Please enter at least 2-3 words.")

    else:

        with st.spinner("🤖 Processing Message..."):
            time.sleep(1)

        input_label = vectorizer.transform([sms])
        prediction = model.predict(input_label)[0]

        if prediction == 0:

            st.success("✅ HAM MESSAGE")

            st.write("### 🌟 Advantages")
            st.write("✔ Your text appears safe")
            st.write("✔ Trusted Sender Content")
            st.write("✔ No Suspicious Links Detected")
            st.write("✔ Low Security Risk")
            st.write("✔ Safe for Communication")

        else:

            st.error("🚨 SPAM MESSAGE")

            st.write("### 🛡 Security Precautions")
            st.write("⚠ Do Not Click Unknown Links")
            st.write("⚠ Never Share OTP")
            st.write("⚠ Never Share Passwords")
            st.write("⚠ Verify Sender Identity")
            st.write("⚠ Report Suspicious Messages")

# Footer
st.markdown("---")
st.caption("SMS Spam Detection System | Naive Bayes Classifier")
