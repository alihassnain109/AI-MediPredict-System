import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("medical_dataset.csv")

# Encode gender
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Encode disease
le = LabelEncoder()
df['disease'] = le.fit_transform(df['disease'])

X = df.drop("disease", axis=1)
y = df["disease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Model trained successfully ✅")
