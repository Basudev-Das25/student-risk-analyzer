import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Load dataset
data = pd.read_csv("../data/student_risk.csv")

#separate feature and target
X = data.drop("risk_level", axis=1)
Y = data["risk_level"]

# print("Feature sample:")
# print(Y.head())

#Encode target labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

#Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y_encoded,
    test_size=0.4,
    random_state=42,
    stratify=Y_encoded
)

#train Random forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, Y_train)


#Predictions
Y_pred = model.predict(X_test)

#Evaluation
print("Classification report:")
print(classification_report(Y_test, Y_pred, target_names=label_encoder.classes_))

print("\nConfussion Matrix")
print(confusion_matrix(Y_test, Y_pred))

#feature importance analysis
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)


import pickle
import os

#Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#path to app directory
APP_DIR = os.path.join(BASE_DIR, "app")
os.makedirs(APP_DIR, exist_ok=True)

#Save trained model
MODEL_PATH = os.path.join(APP_DIR, "risk_model.pkl")
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

#Save label encoder 
ENCODER_PATH = os.path.join(APP_DIR, "label_encoder.pkl")  
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)


print("\nModel and labelEncoder saved successfully")