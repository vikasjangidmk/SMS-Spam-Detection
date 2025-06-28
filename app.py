import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model
model = joblib.load('model.pkl')

# Preprocessing function
def preprocess_text(text):
    lm = WordNetLemmatizer()
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [word for word in review if word not in stopwords.words('english')]
    review = [lm.lemmatize(word) for word in review]
    return " ".join(review)

# Streamlit UI
st.title("📩 SMS Spam Detection")
st.write("This app uses a trained Machine Learning model to classify SMS messages as *Spam* or *Ham (Not Spam)*.")

# Text input
user_input = st.text_area("Enter the SMS message:", "")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        processed_input = preprocess_text(user_input)
        prediction = model.predict([processed_input])[0]
        if prediction == 'spam':
            st.error("🚫 This message is classified as *SPAM*.")
        else:
            st.success("✅ This message is classified as *HAM* (not spam).")