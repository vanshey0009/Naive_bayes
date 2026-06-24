import streamlit as st
import pickle
import time

# Load model and vectorizer
with open("spam_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="SMS Spam Detection System",
    page_icon="📧",
    layout="centered"
)

# Header Section
st.markdown("""
<div style="
width:80%;
margin:auto;
padding:30px;
text-align:center;
background:linear-gradient(135deg,#4facfe,#00f2fe);
color:white;
border-radius:20px;
font-family:Arial;
">

<h1>📧 SMS Spam Detection System</h1>

<h3>Machine Learning Based Text Analytics</h3>

<div style="
background:white;
color:black;
padding:15px;
border-radius:15px;
">

📌 Instructions

<br><br>

📩 Enter your SMS below

🚪 Type <b>exit</b> to close system <br>

🔒 Never share sensitive information online

</div>

</div>
""", unsafe_allow_html=True)

st.write("")

# Input
sms = st.text_area(
    "📩 Enter Message",
    height=150,
    placeholder="Type your SMS here..."
)

# Exit Option
if sms.lower().strip() == "exit":
    st.success("👋 Thank You! Session Ended Successfully")
    st.stop()

# Analyze Button
if st.button("🔍 Analyze Message"):

    if not sms.strip():
        st.warning("Please enter a message.")

    else:

        with st.spinner("🤖 Processing Message..."):
            time.sleep(1)

        input_label = vectorizer.transform([sms])
        prediction = clf.predict(input_label)[0]

        # Message Too Short
        if len(sms.split()) < 2:

            result = "⚠ MESSAGE TOO SHORT"
            color = "#fd7e14"

            info = """
            ⚠ Short Message Warning

            Please enter a longer message for accurate prediction.

            ✔ Minimum 2-3 words recommended
            ✔ More text improves prediction accuracy
            ✔ Try entering a complete sentence
            """

        else:

            if prediction == 0:

                result = "✅ HAM MESSAGE"
                color = "green"

                info = """
                🌟 Advantages

                ✔ Your text appears safe
                ✔ Trusted Sender Content
                ✔ No Suspicious Links Detected
                ✔ Low Security Risk
                ✔ Safe for Communication
                """

            else:

                result = "🚨 SPAM MESSAGE"
                color = "red"

                info = """
                🛡 Security Precautions

                ⚠ Do Not Click Unknown Links
                ⚠ Never Share OTP
                ⚠ Never Share Passwords
                ⚠ Verify Sender Identity
                ⚠ Report Suspicious Messages
                """

        # Result Card
        st.markdown(f"""
        <div style="
        width:80%;
        margin:auto;
        padding:30px;
        border-radius:20px;
        border:4px solid {color};
        font-family:Arial;
        background:white;
        ">

        <h1 style="color:{color};text-align:center;">
        {result}
        </h1>

        <hr>

        <h3>📨 Message</h3>

        <p>{sms}</p>

        <h3>📋 THINGS YOU SHOULD KNOW ABOUT THIS TEXT</h3>

        {info}

        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<center><h4>📧 SMS Spam Detection System</h4></center>",
    unsafe_allow_html=True
)
