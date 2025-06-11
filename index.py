import streamlit as st
import pandas as pd
import pickle
import re
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Set page layout to wide for larger tabs
st.set_page_config(layout="wide", page_title="News Classifier", page_icon="📰")  # Enhanced layout

# Load models and vectorizer
with open('log_reg.pkl', 'rb') as file:
    log_reg = pickle.load(file)

with open('svc_model.pkl', 'rb') as file:
    svc_model = pickle.load(file)

with open('rf_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

with open('nb_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocessing function
def preprocess_text(content):
    words = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    return ' '.join(words)

# Load dataset for metrics
@st.cache_data
def load_data():
    news_data = pd.read_csv('train.csv')
    news_data = news_data.fillna(' ')
    news_data['title'] = news_data['title'].apply(preprocess_text)
    X = vectorizer.transform(news_data['title'])
    y = news_data['label']
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

X_train, X_test, Y_train, Y_test = load_data()

# Evaluate model function
def evaluate_model(model, model_name):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    accuracy_train = accuracy_score(Y_train, train_pred)
    accuracy_test = accuracy_score(Y_test, test_pred)
    class_rep = classification_report(Y_test, test_pred, output_dict=True)
    conf_matrix = confusion_matrix(Y_test, test_pred)

    st.markdown(f"<h3 style='color: #007BFF;'>{model_name} Metrics</h3>", unsafe_allow_html=True)
    st.markdown(f"""
    - <b>Training Accuracy:</b> {accuracy_train:.2f}  
    - <b>Testing Accuracy:</b> {accuracy_test:.2f}  
    """, unsafe_allow_html=True)

    st.markdown("<h4 style='color: #28a745;'>Classification Report:</h4>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(class_rep).transpose())

    st.markdown("<h4 style='color: #dc3545;'>Confusion Matrix:</h4>", unsafe_allow_html=True)
    st.write(conf_matrix)

# Predict function
def predict(news_title, model):
    preprocessed_title = preprocess_text(news_title)
    title_vector = vectorizer.transform([preprocessed_title])
    prediction = model.predict(title_vector)[0]
    return "Real" if prediction == 1 else "Fake"

# Streamlit app layout
st.markdown(
    """
    <style>
        body {
            color: #333;
        }
        .main {
            padding-top: 20px;
            padding-left: 30px;
            padding-right: 30px;
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        .stTextInput input {
            padding: 10px;
            font-size: 18px;
        }
        .stButton button {
            background-color: #007BFF;
            color: white;
            padding: 10px 20px;
            font-size: 18px;
            border-radius: 5px;
            border: 1px solid #007BFF;
        }
        .stButton button:hover {
            background-color: #0056b3;
        }
        .stSelectbox div {
            font-size: 18px;
        }
        .stTabs {
            font-size: 18px;
        }
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #007BFF;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        h3 {
            color: #007BFF;
        }
        .output-card {
            background-color: #e7f3fe;
            border-left: 5px solid #007BFF;
            border-radius: 5px;
            padding: 15px;
            margin-top: 20px;
        }
        .output-card h2 {
            color: #007BFF;
            margin: 0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 News Classification: Real or Fake")

# Main tabs
main_tabs = st.tabs(["Model Metrics", "Make Prediction"])

# Metrics Tab
with main_tabs[0]:
    st.header("Model Performance")
    model_tabs = st.tabs(["Random Forest", "Logistic Regression", "Naive Bayes", "Support Vector Machine"])

    # Random Forest Metrics
    with model_tabs[0]:
        evaluate_model(rf_model, "Random Forest")

    # Logistic Regression Metrics
    with model_tabs[1]:
        evaluate_model(log_reg, "Logistic Regression")

    # Naive Bayes Metrics
    with model_tabs[2]:
        evaluate_model(nb_model, "Naive Bayes")

    # Support Vector Machine Metrics
    with model_tabs[3]:
        evaluate_model(svc_model, "Support Vector Machine")

# Prediction Tab
with main_tabs[1]:
    st.header("Make a Prediction")

    # Layout for the prediction section
    col1, col2 = st.columns([2, 3])

    with col1:
        # Input for news title
        news_title = st.text_input("Enter a news title:", key="news_input")

    with col2:
        # Model selection dropdown
        model_name = st.selectbox(
            "Select a model for prediction:",
            ["Random Forest", "Logistic Regression", "Naive Bayes", "Support Vector Machine"]
        )

    if st.button("Predict"):
        if news_title.strip() == "":
            st.error("Please enter a news title.")
        else:
            if model_name == "Random Forest":
                result = predict(news_title, rf_model)
            elif model_name == "Logistic Regression":
                result = predict(news_title, log_reg)
            elif model_name == "Naive Bayes":
                result = predict(news_title, nb_model)
            elif model_name == "Support Vector Machine":
                result = predict(news_title, svc_model)

            st.markdown(
                f"""
                <div class="output-card">
                    <h2>Prediction: {result}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )