import streamlit as st
import pickle
import time

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="SMS Spam Detection", page_icon="📧", layout="centered")

# Header
st.markdown("""
<div style="
padding:30px;
text-align:center;
background:linear-gradient(135deg,#4facfe,#00f2fe);
color:white;
border-radius:20px;
">

<h1>📧 SMS Spam Detection System</h1>
<h3>Machine Learning Based Text Analytics</h3>

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Instructions Card
st.markdown("""
<div style="
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,0.1);
">

<h3>📌 Instructions</h3>

📩 Enter your SMS below<br><br>

🚪 Type <b>exit</b> to quit<br><br>

🔒 Never share sensitive information online

</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Input Box
sms = st.text_area("📩 Enter Message")

if st.button("Predict"):

    with st.spinner("🤖 Processing Message..."):
        time.sleep(1)

    input_label = vectorizer.transform([sms])

    if len(sms.split()) < 2:

        st.warning("⚠ MESSAGE TOO SHORT")

        st.markdown("""
### ⚠ Short Message Warning

✔ Minimum 2-3 words recommended

✔ More text improves prediction accuracy

✔ Try entering a complete sentence
""")

    else:

        prediction = model.predict(input_label)[0]

        if prediction == 0:

            st.success("✅ HAM MESSAGE")

            st.markdown(f"""
### 📨 Message

{sms}

### 🌟 Advantages

✔ Your text appears safe

✔ Trusted Sender Content

✔ No Suspicious Links Detected

✔ Low Security Risk

✔ Safe for Communication
""")

        else:

            st.error("🚨 SPAM MESSAGE")

            st.markdown(f"""
### 📨 Message

{sms}

### 🛡 Security Precautions

⚠ Do Not Click Unknown Links

⚠ Never Share OTP

⚠ Never Share Passwords

⚠ Verify Sender Identity

⚠ Report Suspicious Messages
""")
