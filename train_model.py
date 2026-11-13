"""Train the Titanic survival model and save it as model.joblib.

Run once:  python train_model.py
This is the same model as Project 04 (Decision Tree, max_depth=4).
"""
import joblib
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# The order of these columns MUST match the order used in app.py
FEATURES = ["pclass", "sex", "age", "fare"]

df = sns.load_dataset("titanic")[["survived"] + FEATURES].dropna()
df["sex"] = (df["sex"] == "female").astype(int)   # female = 1, male = 0

X, y = df[FEATURES], df["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

baseline = max(y_test.mean(), 1 - y_test.mean())
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Baseline accuracy: {baseline:.3f}")
print(f"Test accuracy    : {accuracy:.3f}")

joblib.dump(model, "model.joblib")
print("Saved model.joblib with features:", FEATURES)
