import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

print("Extracting advanced physical & temporal features...")
df = pd.read_csv("Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# 1. Timeline & Target Setup
df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
df["closed_datetime"] = pd.to_datetime(df["closed_datetime"], errors="coerce")
df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], errors="coerce")
df["end_time"] = df["closed_datetime"].fillna(df["resolved_datetime"])
df["duration_mins"] = (df["end_time"] - df["start_datetime"]).dt.total_seconds() / 60.0

has_real_duration = df["duration_mins"].notna() & (df["duration_mins"] > 0) & (df["duration_mins"] < 24 * 60 * 3)
df_clean = df[has_real_duration].copy()
df_clean["is_high_impact"] = (df_clean["duration_mins"] > 60).astype(int)

# 2. Extract The Golden Features (Time + Physical Blockage)
df_clean["hour"] = df_clean["start_datetime"].dt.hour.fillna(12).astype(int)
df_clean["day_of_week"] = df_clean["start_datetime"].dt.dayofweek.fillna(0).astype(int) # 0=Mon, 6=Sun

df_clean["veh_type"] = df_clean["veh_type"].fillna("not_applicable").astype(str).str.strip().str.lower()
df_clean["event_type"] = df_clean["event_type"].fillna("unplanned").astype(str).str.strip().str.lower()
df_clean["event_cause"] = df_clean["event_cause"].astype(str).str.strip().str.lower().replace({"debris": "debris"})
df_clean = df_clean[~df_clean["event_cause"].str.contains("test|demo", na=False)]
df_clean["corridor"] = df_clean["corridor"].fillna("Unknown Corridor").astype(str).str.strip()

df_clean["requires_road_closure"] = df_clean["requires_road_closure"].astype(str).str.upper().str.strip()
df_clean["requires_road_closure"] = (df_clean["requires_road_closure"] == "TRUE").astype(int)

# 3. Vectorization Pipeline
encoders = {}
categorical_features = ["event_type", "event_cause", "corridor", "veh_type"]
X_encoded = pd.DataFrame()

for col in categorical_features:
    le = LabelEncoder()
    unique_vals = list(df_clean[col].unique()) + ["unknown", "unseen"]
    le.fit(unique_vals)
    X_encoded[col] = le.transform(df_clean[col])
    encoders[col] = le

X_encoded["requires_road_closure"] = df_clean["requires_road_closure"].values
X_encoded["hour"] = df_clean["hour"].values
X_encoded["day_of_week"] = df_clean["day_of_week"].values
y = df_clean["is_high_impact"].values

# 4. Train the Expanded Model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=300,        
    max_depth=12,            
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=40
)
model.fit(X_train, y_train)
print(f"Real-World Accuracy: {model.score(X_test, y_test):.2%}")

# 5. Export
with open("congestion_model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "encoders": encoders,
        "unique_causes": sorted(df_clean["event_cause"].unique().tolist()),
        "unique_corridors": sorted(df_clean["corridor"].unique().tolist()),
        "unique_veh_types": sorted(df_clean["veh_type"].unique().tolist()),
        "feature_order": ["event_type", "event_cause", "corridor", "veh_type", "requires_road_closure", "hour", "day_of_week"]
    }, f)
print(" Master weights exported to 'congestion_model.pkl'")