import pickle
import os

print("Current directory:", os.getcwd())
print("\nFiles in directory:")
for f in os.listdir('.'):
    print(f"  - {f}")

print("\n--- Testing Model ---")
try:
    model = pickle.load(open('model.pkl', 'rb'))
    print("SUCCESS: Model loaded -", type(model))
except Exception as e:
    print("ERROR loading model:", e)

print("\n--- Testing Vectorizer ---")
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    print("SUCCESS: Vectorizer loaded -", type(vectorizer))
except Exception as e:
    print("ERROR loading vectorizer:", e)