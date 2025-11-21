# create_feature_cols.py
import pandas as pd, json, os
df = pd.read_csv("C:\\Users\\chand\\OneDrive\\Desktop\\ADSAI\\ADSAI\\ADSTRA\\train.csv")
X = df.drop(['Sale_Amount','Ad_ID','Ad_Date'], axis=1, errors='ignore')
features = list(X.columns)
os.makedirs("ml/artifacts", exist_ok=True)
with open(os.path.join("ml","artifacts","feature_columns.json"), "w") as f:
    json.dump(features, f)
print("Saved feature_columns.json with", len(features), "columns -> ml/artifacts/feature_columns.json")
