import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']
print(df.head())
print(df.shape)
print(df.info())

# Train split test
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)
print("Training set size:", X_train.shape)

# Train the model
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
accuracy = (y_pred == y_test).mean()
print("Model accuracy:", accuracy)

# Visualize results
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy', 'Error Rate'], [
        accuracy, 1-accuracy], color=['blue', 'red'])
plt.ylim(0, 1)
plt.title('Model Performance')
plt.ylabel('Proportion')
plt.show()

# Save the model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Model and vectorizer saved.")
