
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Prepare Data
df = pd.read_csv('creditcard.csv')
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))

X = df.drop('Class', axis=1)
y = df['Class']

# Stratify ensures the 0.17% fraud is in both sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Define Individual Models
# We use 'balanced' weights to handle the fraud imbalance automatically
lr = LogisticRegression(class_weight='balanced', max_iter=1000)
dt = DecisionTreeClassifier(criterion='entropy', max_depth=5, class_weight='balanced')
# probability=True is required for 'soft' voting
svm = SVC(probability=True, class_weight='balanced', kernel='rbf')

# 3. Create the Voting Classifier
# This uses all 3 together!
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('dt', dt), ('svm', svm)],
    voting='soft'
)

# 4. Train and Predict
print("Training the ensemble... (This may take a few minutes due to SVM)")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# 5. Final Results
print("\n--- Final Ensemble Results ---")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Create a quick comparison
models = {'Logistic': lr, 'Decision Tree': dt, 'SVM': svm, 'Ensemble': ensemble}

print("\n--- Final Comparison (Recall Score) ---")
for name, model in models.items():
    score = metrics.recall_score(y_test, model.predict(X_test))
    print(f"{name}: {score:.4f}")