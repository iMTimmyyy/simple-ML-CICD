from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import pandas as pd
import skops.io as skio

df = pd.read_csv("./Data/drug.csv").sample(frac=1).reset_index(drop=True)
X = df.drop(columns="Drug").values
y = df["Drug"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
numeric_features = [0, 4]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_features = [1, 2, 3]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder()),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=0)),
    ]
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

with open("./Results/metrics.txt", "w") as f:
    f.write(f"\nAccuracy: {accuracy:.2f}, F1 score: {f1:.2f}")

plot = ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plot.plot()
plt.savefig("./Results/confusion_matrix.png", dpi=120)

skio.dump(clf, "./Model/drug_classifier.skops")
