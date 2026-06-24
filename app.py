import streamlit as st
import pickle
import time

# Load model and vectorizer

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
page_title="SMS Spam Detection",
page_icon="📧",
layout="wide"
)

# Sidebar

st.sidebar.title("📋 Navigation")

page = st.sidebar.radio(
"Select Page",
[
"🏠 Home",
"📊 Model Information",
"📩 Spam Detection",
"👨‍💻 Developer"
]
)

# HOME PAGE

if page == "🏠 Home":


  st.markdown("""
   <div style="
    padding:30px;
    text-align:center;
   background:linear-gradient(135deg,#4facfe,#00f2fe);
    color:white;
    border-radius:20px;">
   <h1>📧 SMS Spam Detection System</h1>
  <h3>Machine Learning Based Text Analytics</h3>
  </div>
  """, unsafe_allow_html=True)

st.write("")

st.info("""
📌 Project Overview

• Detects Spam and Ham SMS Messages

• Uses Machine Learning

• Built using Multinomial Naive Bayes

• Feature Extraction using CountVectorizer

• Fast and Accurate Prediction
""")


# MODEL PAGE

elif page == "📊 Model Information":


   st.title("📊 Model Information")

   st.subheader("Algorithm Used")

  st.code("""


  from sklearn.naive_bayes import MultinomialNB

  clf = MultinomialNB()
   clf.fit(X_train_counts, y_train)
  """)


  st.success("Model : Multinomial Naive Bayes")

st.subheader("Workflow")

st.write("""
1. Load Dataset

2. Clean Text

3. Convert Text to Numbers using CountVectorizer

4. Train Naive Bayes Model

5. Predict Spam/Ham

6. Evaluate Accuracy
""")


# PREDICTION PAGE

elif page == "📩 Spam Detection":


    st.title("📩 SMS Spam Detection")

  sms = st.text_area(
    "Enter SMS Message",
    height=200,
    placeholder="Type your SMS here..."
    )

if st.button("🔍 Analyze Message"):

    if not sms.strip():

        st.warning("Please enter a valid message.")

    else:

        with st.spinner("🤖 Processing Message..."):
            time.sleep(1)

        input_label = vectorizer.transform([sms])

        if len(sms.split()) < 2:

            st.warning("⚠ MESSAGE TOO SHORT")

        else:

            prediction = model.predict(input_label)[0]

            if prediction == 0:

                st.success("✅ HAM MESSAGE")

                st.markdown(f"""
                ### 📨 Message

                {sms}

                ### 🌟 Advantages

                ✔ Trusted Sender Content

                ✔ No Suspicious Links Detected

                ✔ Low Security Risk

                ✔ Safe Communication
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


# DEVELOPER PAGE

elif page == "👨‍💻 collage project":



st.write("Project Name : SMS Spam Detection System")

st.write("Algorithm : Multinomial Naive Bayes")



st.write("Dataset : SMS Spam Collection Dataset")

st.success("Developed as NTCC / PBEL Project")

