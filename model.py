import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def train_and_save():
    data = pd.read_csv("medical_dataset_final_v2_5000.csv")
    data.columns = data.columns.str.strip()

    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    le = LabelEncoder()
    data['disease'] = le.fit_transform(data['disease'])

    X = data.drop("disease", axis=1)
    y = data["disease"]
    
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,      
            max_depth=20,          
            min_samples_split=2,   
            class_weight="balanced", 
            random_state=42
        ))
    ])
    print("Training model... Please wait.")
    model.fit(X, y)

    model_data = {
        "model": model,
        "le": le,
        "symptoms": X.columns.tolist()
    }
    joblib.dump(model_data, "medi_predict_model.joblib")
    print("Model saved successfully as 'medi_predict_model.joblib' ")

if __name__ == "__main__":
    train_and_save()