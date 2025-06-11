import pandas as pd
import re
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
news_data = pd.read_csv('train.csv')
news_data = news_data.fillna(' ')

# Preprocessing
stemmer = PorterStemmer()
def preprocess_text(content):
    words = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    return ' '.join(words)

news_data['title'] = news_data['title'].apply(preprocess_text)

# Features and labels
X = news_data['title']
y = news_data['label']

# Convert to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_data['title'].values)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train models
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

svc_model = SVC()
svc_model.fit(X_train, Y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, Y_train)

# Save models
with open('log_reg.pkl', 'wb') as file:
    pickle.dump(log_reg, file)

with open('svc_model.pkl', 'wb') as file:
    pickle.dump(svc_model, file)

with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

with open('nb_model.pkl', 'wb') as file:
    pickle.dump(nb_model, file)

# Save vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

print("Models and vectorizer have been saved.")
