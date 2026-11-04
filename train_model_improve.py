"""Train the spam classifier and save it as a single sklearn Pipeline.

Output: spam_pipeline.pkl (preprocessing + TF-IDF + Multinomial NB).
"""
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from preprocessing import preprocess_text

MODEL_PATH = 'spam_pipeline.pkl'
RANDOM_STATE = 42

# --- Data Loading ---
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    sys.exit("Error: 'spam.csv' not found. Please ensure the file is in the directory.")

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df = df.dropna()

# Custom list of spam phrases to augment the dataset
additional_spam = [
    "free money click now", "100% free limited time offer", "congratulations you won a prize",
    "click here to claim your reward", "urgent act now before its too late",
    "make money fast working from home", "lose weight quick easy pills",
    "cheap viagra cialis online", "you have been selected for a special offer",
    "exclusive deal just for you", "act fast only a few spots left",
    "guaranteed profit no risk", "double your income in 7 days",
    "earn $500 per day from home", "no experience required start today",
    "winner claim your cash prize now", "limited time discount 70% off",
    "this offer expires tonight", "instant approval no credit check",
    "get rich quick scheme", "risk free investment opportunity",
    "you are pre approved for a loan", "lowest price guaranteed",
    "free trial cancel anytime", "unlock your financial freedom",
    "once in a lifetime opportunity", "make passive income today",
    "don’t miss out on this deal", "secret method to make money",
    "your account has been compromised click here", "verify your account immediately",
    "security alert unusual activity detected", "reset your password now",
    "confirm your identity to avoid suspension", "your payment has failed update info",
    "final notice before account closure", "you received a new voicemail listen now",
    "tax refund waiting for you", "claim your government grant",
    "work from home and earn instantly", "miracle cure doctors hate",
    "burn fat overnight guaranteed", "natural remedy that works instantly",
    "boost testosterone fast", "enlarge size naturally in days",
    "cheap meds without prescription", "order pills discreet shipping",
    "online pharmacy no prescription", "click this link to get your reward",
    "free bitcoin giveaway", "double your crypto instantly",
    "limited airdrop claim now", "send 1 btc get 2 back",
    "investment opportunity in crypto", "nft giveaway exclusive access",
    "hot singles in your area", "meet lonely women tonight",
    "adult content waiting for you", "watch private videos now",
    "dating offer expires today", "you have an unpaid invoice",
    "download your receipt here", "open the attachment immediately",
    "important document attached", "payment confirmation required",
    "special promotion just launched", "huge savings available today",
    "best deal of the year", "only for selected customers",
    "reward points expiring soon", "last chance to save big",
    "exclusive vip access", "free gift included with purchase",
    "instant cash bonus", "money back guarantee",
    "zero cost sign up", "get started in 60 seconds",
    "no hidden fees ever", "free consultation available"
]

print('--- Data Summary (real data) ---')
print(f'Total samples: {len(df)}')
print(f'Ham count: {(df["label"] == 0).sum()}')
print(f'Spam count: {(df["label"] == 1).sum()}')

# --- Train/Test Split ---
# Split the REAL data first. The test set must contain only real messages,
# otherwise the easy synthetic phrases inflate the test metrics.
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df['label'],
)

# Augment the TRAINING set only with the synthetic spam phrases.
X_train = pd.concat([X_train, pd.Series(additional_spam)], ignore_index=True)
y_train = pd.concat([y_train, pd.Series([1] * len(additional_spam))], ignore_index=True)
print(f'Training samples after augmentation: {len(X_train)} (+{len(additional_spam)} synthetic spam)')
print(f'Test samples (real only): {len(X_test)}')

# --- Pipeline ---
# preprocess_text runs inside the vectorizer, so the saved pipeline applies
# exactly the same cleaning at inference time. Passing `preprocessor`
# replaces sklearn's default lowercasing; preprocess_text lowercases itself.
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        preprocessor=preprocess_text,
        max_features=5000,
        ngram_range=(1, 3),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True,
        stop_words='english',
    )),
    ('nb', MultinomialNB(alpha=0.1)),
])

# --- Cross-Validation ---
# The vectorizer is refit inside each fold, so no vocabulary/IDF information
# leaks from the validation fold into training.
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
print(f'\nCross-validation F1 scores: {np.round(cv_scores, 4)}')
print(f'Mean CV F1: {cv_scores.mean():.4f}')

# --- Training and Evaluation ---
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("\n" + "=" * 40)
print("   CLASSIFICATION REPORT (test set)")
print("=" * 40)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam'], digits=4))

# --- Feature Importance: top spam-indicating n-grams ---
vectorizer = pipeline.named_steps['tfidf']
model = pipeline.named_steps['nb']
feature_names = vectorizer.get_feature_names_out()
spam_indicators = model.feature_log_prob_[1] - model.feature_log_prob_[0]

print("\n" + "=" * 40)
print("   TOP 15 SPAM INDICATORS")
print("=" * 40)
top_spam_indices = spam_indicators.argsort()[-15:][::-1]
for i, idx in enumerate(top_spam_indices, 1):
    print(f"{i:2d}. {feature_names[idx]:20s} (Score: {spam_indicators[idx]:+.2f})")

# --- Practical Testing ---
test_samples = [
    "free money click now",
    "Hey, are we still meeting for coffee tomorrow?",
    "WINNER! You've won a £1000 prize. Click here to claim!",
    "I'll be home in 10 minutes, see you then.",
    "100% FREE! Call 555-1234 NOW!!!",
]

print("\n" + "=" * 40)
print("   LIVE PREDICTION TEST")
print("=" * 40)
for text, prob in zip(test_samples, pipeline.predict_proba(test_samples)[:, 1]):
    label = 'Spam' if prob > 0.5 else 'Ham'
    print(f"[{label}] ({prob * 100:5.2f}%) -> {text}")

# --- Saving ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(pipeline, f)
print(f"\nPipeline saved to {MODEL_PATH}.")

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
labels = np.array([
    [f"Correct Ham\n{cm[0, 0]}", f"WRONGLY BLOCKED\n{cm[0, 1]}"],
    [f"Missed Spam\n{cm[1, 0]}", f"Correct Spam\n{cm[1, 1]}"],
])

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=labels, fmt="", cmap='RdYlGn', cbar=False)
plt.title('Spam Filter Performance (test set)')
plt.xlabel('What the Model Decided')
plt.ylabel('What the Message Actually Was')
plt.show()
