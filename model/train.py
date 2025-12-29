import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Load dataset
data = pd.read_csv("../data/student_risk.csv")

#separate feature and target
X = data.drop("risk_level", axis=1)
Y = data["risk_level"]

print("Feature sample:")
print(Y.head())

#Encode target labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

print("\nTarget sample (after encoding):")
print(Y_encoded[:5])

print("\nLabel mapping:")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label} -> {i}")