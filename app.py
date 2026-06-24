import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩"
)

st.title("📩 SMS Spam Detection System")
st.write("Machine Learning Based Text Analytics")

message = st.text_area("Enter your SMS")

if st.button("Predict"):

    if not message.strip():
        st.warning("Please enter a message.")

    elif len(message.split()) < 2:
        st.warning("Message too short. Enter at least 2-3 words.")

    else:
        transformed = vectorizer.transform([message])
        prediction = model.predict(transformed)[0]

        if prediction == 0:
            st.success("✅ HAM MESSAGE")
            st.info("""
Safe message detected.

• Trusted content  
• No suspicious links detected  
• Low security risk
            """)
        else:
            st.error("🚨 SPAM MESSAGE")
            st.warning("""
Possible spam detected.

• Verify sender identity  
• Avoid clicking unknown links  
• Do not share personal information
            """)
