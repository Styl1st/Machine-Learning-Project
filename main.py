import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

test_df = pd.read_csv('Students drugs Addiction Dataset/Student_Drugs_Addiction_Testing_ Dataset/student_addiction_dataset_test.csv')
train_df = pd.read_csv('Students drugs Addiction Dataset/Student_Drugs_Addiction_Training_ Dataset/student_addiction_dataset_train.csv')

# Data Preprocessing
train_df = train_df.dropna()
test_df = test_df.dropna()

train_df["__is_train__"] = 1
test_df["__is_train__"] = 0

full_df = pd.concat([train_df, test_df], ignore_index=True)

categorical_cols = full_df.select_dtypes(include="object").columns

for col in categorical_cols:
    if col == "__is_train__":
        continue
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col])

train_df = full_df[full_df["__is_train__"] == 1].drop(columns="__is_train__")
test_df = full_df[full_df["__is_train__"] == 0].drop(columns="__is_train__")

print(train_df)
print(test_df)

# Chi-2 (X²) Feature Selection
from scipy.stats import chi2_contingency

target_col = "Addiction_Class"

# Run Chi-2 test for each feature against the target
chi2_results = []

for col in train_df.columns:
    if col == target_col:
        continue
    contingency_table = pd.crosstab(train_df[col], train_df[target_col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    chi2_results.append([col, chi2, p])

chi2_df = pd.DataFrame(chi2_results, columns=["Feature", "Chi2", "p-value"])
chi2_df = chi2_df.sort_values(by="Chi2", ascending=False)
print(chi2_df)
print("\n")

# KNN Classifier 
X_train = train_df.drop(columns=["Addiction_Class"])
y_train = train_df["Addiction_Class"]

X_test = test_df.drop(columns=["Addiction_Class"])
y_test = test_df["Addiction_Class"]

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

results_df_KNN = pd.DataFrame({
    "Accuracy": [accuracy_score(y_test, y_pred)],
    "Precision": [precision_score(y_test, y_pred, average='weighted', zero_division=0)],
    "Recall": [recall_score(y_test, y_pred, average='weighted')],
    "F1": [f1_score(y_test, y_pred, average='weighted')]
})
results_df_KNN.index = ["KNeighborsClassifier"]

print(results_df_KNN)


# Correlation Analysis
corr_matrix = train_df.corr()

import seaborn as sns

plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()