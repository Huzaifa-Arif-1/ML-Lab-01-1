from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)
df = iris.frame.copy()

print("Shape:", df.shape)
print(df.head().to_string())
print("Missing values:\n", df.isna().sum())
print("Class counts:\n", df["target"].value_counts())
print("Summary:\n", df.describe().round(2))

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
print("Train/test:", X_train.shape, X_test.shape)

df.iloc[:, :4].hist(figsize=(8, 6))
plt.tight_layout(); plt.savefig("week01_hist.png", dpi=150)
print("Test class counts:\n", y_test.value_counts().sort_index())
