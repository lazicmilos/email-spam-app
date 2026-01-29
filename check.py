import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['label'].value_counts())
print("\nPercentage:")
print(df['label'].value_counts(normalize=True) * 100)

print("\n--- Sample SPAM emails ---")
spam_samples = df[df['label'] == 'spam']['text'].head(3)
for i, text in enumerate(spam_samples, 1):
    print(f"{i}. {text}")

print("\n--- Sample HAM emails ---")
ham_samples = df[df['label'] == 'ham']['text'].head(3)
for i, text in enumerate(ham_samples, 1):
    print(f"{i}. {text}")