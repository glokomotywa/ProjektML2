import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# 1. Wczytanie danych
X, y = load_breast_cancer(return_X_y=True)

# 2. Podział danych
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3. Model: Drzewo decyzyjne
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)[:, 1]

# 4. Model: k-NN (z normalizacją)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

# 5. Metryki
acc_dt = accuracy_score(y_test, y_pred_dt)
auc_dt = roc_auc_score(y_test, y_prob_dt)
acc_knn = accuracy_score(y_test, y_pred_knn)
auc_knn = roc_auc_score(y_test, y_prob_knn)

metrics_df = pd.DataFrame(
    {
        "Model": ["Drzewo decyzyjne", "k-NN"],
        "Accuracy": [acc_dt, acc_knn],
        "ROC AUC": [auc_dt, auc_knn],
    }
)

print("Porównanie metryk:\n", metrics_df.to_string(index=False))

# 6. Krzywa ROC
plt.figure()
RocCurveDisplay.from_predictions(
    y_test, y_prob_dt, name="Drzewo decyzyjne"
)
RocCurveDisplay.from_predictions(
    y_test, y_prob_knn, name="k-NN"
)
plt.title("Krzywe ROC dla obu modeli")
plt.show()

# 7. Wykres słupkowy metryk
plt.figure()
x = np.arange(len(metrics_df["Model"]))
width = 0.35
plt.bar(x - width / 2, metrics_df["Accuracy"], width, label="Accuracy")
plt.bar(x + width / 2, metrics_df["ROC AUC"], width, label="ROC AUC")
plt.xticks(x, metrics_df["Model"])
plt.ylabel("Wartość metryki")
plt.title("Porównanie Accuracy i ROC AUC")
plt.ylim(0, 1.05)
plt.legend()
plt.show()
