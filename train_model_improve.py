import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import re 
import seaborn as sns
import matplotlib.pyplot as plt

# --- Data Loading ---
# Load dataset with specific encoding for SMS spam data
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure the file is in the directory.")
    exit()

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

# Create DataFrame for additional spam and merge
# Fixed: We use 'spam' string here to match original labels before mapping
additional_df = pd.DataFrame({
    'text': additional_spam,
    'label': ['spam'] * len(additional_spam)
})
df = pd.concat([df, additional_df], ignore_index=True)

# --- Preprocessing ---
def preprocess_text(text):
    """
    Cleans text by lowering case, removing non-alphanumeric characters 
    (except currency and basic punctuation), and removing extra spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Keep letters, numbers, and common spam-related symbols
    text = re.sub(r'[^a-z0-9\s£$€!?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing to text data
df['text'] = df['text'].apply(preprocess_text)

# Map labels: ham -> 0, spam -> 1
# This must happen AFTER merging to avoid NaN issues
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any rows that might have become empty or NaN after processing
df.dropna(inplace=True)

print('--- Data Summary ---')
print(f'Total samples: {len(df)}')
print(f'Ham count: {df["label"].value_counts()[0]}')
print(f'Spam count: {df["label"].value_counts()[1]}')

# --- Model Preparation ---
# Split data using stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], 
    test_size=0.2, 
    random_state=42, 
    stratify=df['label']
)

# Vectorize text using TF-IDF   
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.85,
    sublinear_tf=True,
    stop_words='english' # Optional: removes common words like 'the', 'is'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# --- Training and Evaluation ---
# Use Multinomial Naive Bayes with small alpha for smoothing
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vec, y_train)

# Perform Cross-Validation to check for overfitting
cv_score = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1')
print(f'\nCross-validation F1 scores: {cv_score}')
print(f'Mean CV F1: {cv_score.mean():.4f}')

# Final evaluation on the test set
y_pred = model.predict(X_test_vec)

print("\n" + "="*40)
print("   CLASSIFICATION REPORT")
print("="*40)
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Feature Importance: Identifying top spam-triggering words
feature_names = vectorizer.get_feature_names_out()
spam_log_probs = model.feature_log_prob_[1]
ham_log_probs = model.feature_log_prob_[0]
spam_indicators = spam_log_probs - ham_log_probs

print("\n" + "="*40)
print("   TOP 15 SPAM INDICATORS")
print("="*40)
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

print("\n" + "="*40)
print("   LIVE PREDICTION TEST")
print("="*40)
for text in test_samples:
    proc = preprocess_text(text)
    vec = vectorizer.transform([proc])
    prob = model.predict_proba(vec)[0, 1]
    label = 'Spam' if prob > 0.5 else 'Ham'
    print(f"[{label}] ({prob*100:5.2f}%) -> {text}")

# --- Saving the Artifacts ---
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully.")

# Visualizing Confusion Matrix in procentages
cm = confusion_matrix(y_test, y_pred)
labels = np.array([
    [f"Correct Ham\n{cm[0,0]}", f"WRONGLY BLOCKED\n{cm[0,1]}"],
    [f"Missed Spam\n{cm[1,0]}", f"Correct Spam\n{cm[1,1]}"]
])

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=labels, fmt="", cmap='RdYlGn', cbar=False)

plt.title('Spam Filter Performance - Detailed View')
plt.xlabel('What the Model Decided')
plt.ylabel('What the Email Actually Was')
plt.show()

#End of file 