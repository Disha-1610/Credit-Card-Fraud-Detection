import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Load dataset
df = pd.read_csv("creditcard.csv")

# Scale features
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1,1))

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Models
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
dt = DecisionTreeClassifier(criterion="entropy", max_depth=5, class_weight="balanced")
svm = SVC(probability=True, kernel="rbf", class_weight="balanced")

ensemble = VotingClassifier(
    estimators=[("lr", lr), ("dt", dt), ("svm", svm)],
    voting="soft"
)

print("Training model...")
ensemble.fit(X_train, y_train)

# Save model + scaler
joblib.dump(ensemble, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model saved!")