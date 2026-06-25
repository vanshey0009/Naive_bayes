# Naive_bayes
# 📧 SMS Spam Detection System using Naive Bayes

A Machine Learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using the **Multinomial Naive Bayes Algorithm**. The project includes data preprocessing, feature extraction, model training, evaluation, visualizations, and an interactive prediction interface. 

---

##  Project Overview

Spam messages are unwanted messages that often contain advertisements, scams, or fraudulent links. This project uses Natural Language Processing (NLP) and Machine Learning techniques to automatically detect spam messages.

The model is trained using the **Naive Bayes Classification Algorithm**, which is highly effective for text classification tasks.

---

##  Features

- SMS Spam Detection using Machine Learning
- Text Preprocessing
- Count Vectorization (Bag of Words)
- Multinomial Naive Bayes Classifier
- Model Accuracy Evaluation
- Confusion Matrix Visualization
- Spam Word Cloud
- Error Analysis
- Message Length Analysis
- Interactive User Input Interface
- Real-Time Spam Prediction

---


## Step 1: Import Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
```

### Explanation

* **Pandas** → Data handling and analysis.
* **NumPy** → Numerical operations.
* **Matplotlib & Seaborn** → Data visualization.
* **CountVectorizer** → Converts text into numerical vectors.
* **MultinomialNB** → Naive Bayes algorithm for text classification.
* **Metrics** → Evaluate model performance.

---

## Step 2: Load Dataset

```python
df = pd.read_csv("train.csv", encoding="latin-1")
```

Explanation

Loads the SMS dataset into a DataFrame.

---

## Step 3: View Dataset

```python
df.head()
```

 Explanation

Displays the first five records of the dataset.

---

## Step 4: Data Cleaning

```python
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
```

Explanation

* Keep only required columns.
* Rename columns for better readability.

---

## Step 5: Check Missing Values

```python
df.isnull().sum()
```

 Explanation

Checks whether any rows contain missing values.

---

## Step 6: Convert Labels into Numbers

```python
df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})
```

Explanation

Machine Learning models work with numbers, so:

* Ham → 0
* Spam → 1

---

## Step 7: Visualize Dataset Distribution

```python
df['label'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%'
)
plt.show()
```

 Explanation

Shows percentage of spam and ham messages.


<img width="513" height="471" alt="Screenshot 2026-06-19 102241" src="https://github.com/user-attachments/assets/92a08aa1-24f9-47fa-816c-8ff5712609a7" />

---

## Step 8: Split Data

```python
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

 Explanation

* 80% Training Data
* 20% Testing Data

---

## Step 9: Convert Text into Numbers

```python
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

 Explanation

Transforms text messages into numerical vectors using Bag of Words.

Example:

Message:

```
Free money now
```

Becomes:

```
[1,1,1]
```

---

## Step 10: Train Naive Bayes Model

```python
model = MultinomialNB()
model.fit(X_train_vec, y_train)
```

 Explanation

The model learns patterns from training messages.

---

## Step 11: Make Predictions

```python
predictions = model.predict(X_test_vec)
```

 Explanation

Predicts whether messages are spam or ham.

---

## Step 12: Calculate Accuracy

```python
accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)
```

 Explanation

Measures how many predictions are correct.


<img width="565" height="260" alt="image" src="https://github.com/user-attachments/assets/a4c3b49c-1bf3-4390-8fa9-f5609cfe795b" />

---

## Step 13: Generate Confusion Matrix

```python
cm = confusion_matrix(
    y_test,
    predictions
)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.show()
```

 Explanation

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

<img width="536" height="535" alt="Screenshot 2026-06-19 102811" src="https://github.com/user-attachments/assets/da04350f-73d4-494f-92d8-20dd30d3e271" />


---

## Step 14: Create Spam Word Cloud

```python
from wordcloud import WordCloud

spam_words = " ".join(
    df[df['label']==1]['message']
)

wordcloud = WordCloud(
    width=800,
    height=400
).generate(spam_words)

plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

 Explanation

Displays the most common words found in spam messages.


<img width="1116" height="447" alt="image" src="https://github.com/user-attachments/assets/32e028ae-7883-4270-a0f5-dda932093f78" />


---

 ## Step 15: Real-Time Prediction

```python



from IPython.display import HTML, display, clear_output

import time

# Header
display(HTML("""
<div style="
padding:30px;
text-align:center;
background:linear-gradient(135deg,#4facfe,#00f2fe);
color:white;
border-radius:20px;
font-family:Arial;
margin-bottom:20px;
">
<h1>📧 SMS Spam Detection System</h1>
<h3>Machine Learning Based Text Analytics</h3>
</div>
"""))

# Instructions
display(HTML("""
<div style="
background:#eaf6ff;
padding:15px;
border-left:6px solid #2196F3;
border-radius:10px;
font-family:Arial;
margin-bottom:20px;
">
<h3>📌 Instructions</h3>
<p>📩 Enter your SMS message below</p>
<p>🚪 Type <b>exit</b> to quit</p>
<p>🔒 Never share sensitive information online</p>
</div>
"""))

while True:

    sms = input("📩 Enter Message: ")

    if sms.lower().strip() == "exit":

        display(HTML("""
        <div style="
        background:#d4edda;
        color:#155724;
        padding:15px;
        border-radius:10px;
        margin-top:15px;
        font-family:Arial;
        ">
        👋 Thank you for using the SMS Spam Detection System
        </div>
        """))

        break

    if not sms.strip():

        display(HTML("""
        <div style="
        background:#fff3cd;
        color:#856404;
        padding:15px;
        border-radius:10px;
        margin-top:15px;
        font-family:Arial;
        ">
        ⚠ Please enter a valid message.
        </div>
        """))

        continue

    display(HTML("""
    <div style="
    background:#f8f9fa;
    padding:15px;
    border-radius:10px;
    margin-top:15px;
    font-family:Arial;
    ">
    🤖 Processing Message...
    </div>
    """))

    time.sleep(1)

    input_label = vectorizer.transform([sms])
    prediction = clf.predict(input_label)[0]

    if len(sms.split()) < 2:

        display(HTML("""
        <div style="
        background:#fff3cd;
        color:#856404;
        padding:20px;
        border-radius:10px;
        margin-top:15px;
        font-family:Arial;
        ">
        <h3>⚠ Message Too Short</h3>

        <h4>Message Recommendation</h4>

        ✔ Enter at least 2–3 words<br><br>

        ✔ Longer messages improve prediction accuracy<br><br>

        ✔ Use a complete sentence whenever possible
        </div>
        """))

    else:

        if prediction == 0:

            display(HTML(f"""
            <div style="
            background:#d4edda;
            color:#155724;
            padding:20px;
            border-radius:10px;
            margin-top:15px;
            font-family:Arial;
            ">
            <h2>✅ HAM MESSAGE</h2>

            <h3>📨 Message</h3>

            <p>{sms}</p>

            <h3>📊 Message Analysis Report</h3>

            ✔ Safe message detected<br><br>

            ✔ Trusted sender content<br><br>

            ✔ No suspicious links detected<br><br>

            ✔ Low security risk<br><br>

            ✔ Safe for communication<br><br>

            ✔ Normal conversational text
            </div>
            """))

        else:

            display(HTML(f"""
            <div style="
            background:#f8d7da;
            color:#721c24;
            padding:20px;
            border-radius:10px;
            margin-top:15px;
            font-family:Arial;
            ">
            <h2>🚨 SPAM MESSAGE</h2>

            <h3>📨 Message</h3>

            <p>{sms}</p>

            <h3>🛡 Security Recommendations</h3>

            ⚠ Do not click unknown links<br><br>

            ⚠ Never share OTPs<br><br>

            ⚠ Never share passwords<br><br>

            ⚠ Verify the sender's identity<br><br>

            ⚠ Report suspicious messages
            </div>
            """))

display(HTML("""
<div style="
text-align:center;
margin-top:20px;
font-family:Arial;
color:gray;
">
SMS Spam Detection System | Machine Learning Project using Naive Bayes
</div>
"""))
```

 Explanation

Allows users to test messages in real time.

Example:

Input:

```
Congratulations! You won ₹5000.
```

Output:

```
🚨 SPAM MESSAGE
```

Input:

```
Let's meet tomorrow.
```

Output:

```
✅ HAM MESSAGE
```

---

## Step 16: Project Workflow Summary

```text
Dataset
   ↓
Data Cleaning
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
CountVectorizer
   ↓
Naive Bayes Training
   ↓
Prediction
   ↓
Accuracy Evaluation
   ↓
Visualization
   ↓
Real-Time Spam Detection
```

This workflow demonstrates the complete Machine Learning pipeline for SMS Spam Detection using Natural Language Processing (NLP) and the Multinomial Naive Bayes algorithm.


### project link 
https://naivebayes-6lpwlj6ouxccek5cbhkz3m.streamlit.app/

### student 
Name: vanshey khari 

corse:	B.Tech CSE  (IoT and Cyber Sec. inc BCT)

collage : Amity University Uttar Pradesh, Noida

