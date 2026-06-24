import streamlit as st
import pickle
import time

# Load Model
with open("spam_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page Settings
st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# Header
st.markdown("""
<div style="
padding:30px;
text-align:center;
background:linear-gradient(135deg,#4facfe,#00f2fe);
color:white;
border-radius:20px;
font-family:Arial;
">

<h1>📧 SMS Spam Detection System</h1>

<h3>Machine Learning Based Text Analytics</h3>

</div>
""", unsafe_allow_html=True)

st.write("")

# Instructions
st.info("""
📌 Instructions

📩 Enter your SMS message below

🚪 Type 'exit' to quite

🔒 Never share sensitive information online
""")

# Input Box
sms = st.text_area(
    "📩 Enter Message",
    height=150,
    placeholder="Type your SMS here..."
)

# Exit Option
if sms.lower().strip() == "exit":
    st.success("👋 Thank you for using the SMS Spam Detection System")
    st.stop()

# Prediction Button
if st.button("🔍 Analyze Message"):

    if not sms.strip():

        st.warning("Please enter a valid message.")

    else:

        with st.spinner("🤖 Processing Message..."):
            time.sleep(1)

        input_label = vectorizer.transform([sms])
        prediction = clf.predict(input_label)[0]

        if len(sms.split()) < 2:

            st.warning("⚠ Message Too Short")

            st.markdown("""
### Message Recommendation

✔ Enter at least 2–3 words

✔ Longer messages improve prediction accuracy

✔ Use a complete sentence whenever possible
""")

        else:

            if prediction == 0:

                st.success("✅ HAM MESSAGE")

                st.markdown(f"""
### 📨 Message

{sms}

### 📊 Message Analysis Report

✔ Safe message detected

✔ Trusted sender content

✔ No suspicious links detected

✔ Low security risk

✔ Safe for communication

✔ Normal conversational text
""")

            else:

                st.error("🚨 SPAM MESSAGE")

                st.markdown(f"""
### 📨 Message

{sms}

### 🛡 Security Recommendations

⚠ Do not click unknown links

⚠ Never share OTPs

⚠ Never share passwords

⚠ Verify the sender's identity

⚠ Report suspicious messages
""")

st.write("")

st.caption("SMS Spam Detection System | Machine Learning Project using Naive Bayes")
