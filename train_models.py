# ML Assignment 2 - training script
# Dataset: Bank Marketing (UCI), bank.csv - 4521 rows, 16 features
# Goal: predict whether a client subscribes to a term deposit (column y: yes/no)
#
# What this script does:
#   - splits the data 80/20 (stratified, since only ~11.5% are "yes")
#   - saves the 20% part as test_data.csv so it can be uploaded in the app
#   - trains the 5 required models and prints all 6 metrics for each
#   - saves each trained model in the model/ folder

import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

folder = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(folder, "data", "bank.csv")
model_folder = os.path.join(folder, "model")
test_file = os.path.join(folder, "test_data.csv")
os.makedirs(model_folder, exist_ok=True)

# fixed seed so I get the same split and results every run
seed = 42

# ---- load data ----
print("Loading bank.csv ...")
df = pd.read_csv(data_file, sep=";")          # the UCI file uses ; not ,
print("Rows, columns:", df.shape)

X = df.drop(columns=["y"])
y = (df["y"] == "yes").astype(int)              # yes = 1, no = 0

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# ---- train / test split ----
# stratify keeps the same yes/no ratio in both parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# save the test part with the original yes/no target for the Streamlit app
test_df = X_test.copy()
test_df["y"] = df.loc[X_test.index, "y"]
test_df.to_csv(test_file, index=False)
print("Saved test_data.csv")

# ---- preprocessing ----
# scale the numbers (KNN and SVM need this) and one-hot encode the text columns
prep = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

# the 5 models from the assignment
# DT depth 6 and k=7 for KNN - picked to avoid overfitting, not heavily tuned
models = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed),
                            "logistic_regression.joblib"),
    "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=seed),
                      "decision_tree.joblib"),
    "KNN": (KNeighborsClassifier(n_neighbors=7), "knn.joblib"),
    "Naive Bayes": (GaussianNB(), "naive_bayes.joblib"),
    "Support Vector Machine": (SVC(kernel="rbf", probability=True, random_state=seed),
                               "svm.joblib"),
}

# ---- train and evaluate ----
results = []
for name, (clf, fname) in models.items():
    start = time.time()

    # preprocessing + model saved together, so the app can use raw CSV columns
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]     # needed for AUC

    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })

    joblib.dump(pipe, os.path.join(model_folder, fname))
    print(f"{name}: trained in {time.time() - start:.2f}s, saved as model/{fname}")

# ---- results ----
table = pd.DataFrame(results).round(4)
table.to_csv(os.path.join(model_folder, "metrics.csv"), index=False)

print("\nResults on test data:")
print(table.to_string(index=False))
