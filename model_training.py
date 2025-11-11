import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


# Load dataset
df = pd.read_csv("laundry_usage.csv")


# Preprocessing
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%H:%M", errors="coerce")
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Day"].astype(str)

# High-demand days
df["High_Demand_Day"] = df["DayOfWeek"].apply(lambda x: 1 if x in ["Wed","Sat","Sun"] else 0)

# Label encoders
le_day = LabelEncoder()
le_machine = LabelEncoder()
le_type = LabelEncoder()

df["DayOfWeek_encoded"] = le_day.fit_transform(df["DayOfWeek"])
df["Machine_ID_encoded"] = le_machine.fit_transform(df["Machine_ID"])
df["MachineType_encoded"] = le_type.fit_transform(df["MachineType"])

# Status encoding
df["Status"] = df["Status"].map({"Running": 1, "Idle": 0})

# Multi-output targets
df["Washer_Status"] = df.apply(lambda r: r["Status"] if r["MachineType"]=="Washer" else 0, axis=1)
df["Dryer_Status"] = df.apply(lambda r: r["Status"] if r["MachineType"]=="Dryer" else 0, axis=1)

# Features & target
features = ["Machine_ID_encoded", "MachineType_encoded", "DayOfWeek_encoded", "Hour", "Duration(min)", "High_Demand_Day"]
X = df[features].astype(float)
y = df[["Washer_Status","Dryer_Status"]]

# Scale numeric features
scaler = MinMaxScaler()
X[["Hour","Duration(min)"]] = scaler.fit_transform(X[["Hour","Duration(min)"]])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Multi-Output Random Forest
multi_rf = MultiOutputClassifier(
    RandomForestClassifier(n_estimators=150, max_depth=None, random_state=42, n_jobs=-1)
)
multi_rf.fit(X_train, y_train)

# Predictions
y_pred = multi_rf.predict(X_test)


# Evaluation
def evaluate_model(y_test, y_pred, machine_name):
    acc = accuracy_score(y_test, y_pred)
    print(f"{machine_name} Accuracy: {acc*100:.2f}%")
    print(f"Classification Report ({machine_name}):")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=["Idle","Running"], yticklabels=["Idle","Running"])
    plt.title(f"{machine_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    # Distribution plot
    plt.figure(figsize=(6,4))
    sns.countplot(x=pd.Series(y_pred).replace({1:"Running",0:"Idle"}), palette={"Running":"#3498db","Idle":"#f39c12"})
    plt.title(f"Predicted {machine_name} Status Distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Side-by-side comparison for Washer
washer_comparison = pd.DataFrame({
    "Status": ["Idle", "Running"],
    "Actual": y_test["Washer_Status"].value_counts().reindex([0,1], fill_value=0),
    "Predicted": pd.Series(y_pred[:,0]).value_counts().reindex([0,1], fill_value=0)
})

washer_melted = washer_comparison.melt(id_vars="Status", value_vars=["Actual","Predicted"],var_name="Category", value_name="Count")

plt.figure(figsize=(7,5))
sns.barplot(data=washer_melted, x="Status", y="Count", hue="Category", palette={"Actual":"#3498db","Predicted":"#e67e22"})
plt.title("Washer: Actual vs Predicted Status Counts")
plt.xlabel("Machine Status")
plt.ylabel("Count")
plt.legend(title="Source")
plt.tight_layout()
plt.show()



# Side-by-side comparison for Dryer
dryer_comparison = pd.DataFrame({
    "Status": ["Idle", "Running"],
    "Actual": y_test["Dryer_Status"].value_counts().reindex([0,1], fill_value=0),
    "Predicted": pd.Series(y_pred[:,1]).value_counts().reindex([0,1], fill_value=0)
})

dryer_melted = dryer_comparison.melt(id_vars="Status", value_vars=["Actual","Predicted"],var_name="Category", value_name="Count")

plt.figure(figsize=(7,5))
sns.barplot(data=dryer_melted, x="Status", y="Count", hue="Category", palette={"Actual":"#3498db","Predicted":"#e67e22"})
plt.title("Dryer: Actual vs Predicted Status Counts")
plt.xlabel("Machine Status")
plt.ylabel("Count")
plt.legend(title="Source")
plt.tight_layout()
plt.show()

# Evaluate Washer and Dryer separately
evaluate_model(y_test["Washer_Status"], y_pred[:,0], "Washer")
evaluate_model(y_test["Dryer_Status"], y_pred[:,1], "Dryer")


# Save model and encoders
joblib.dump(multi_rf, "laundry_status_multi_model.pkl")
joblib.dump(le_day, "le_day.pkl")
joblib.dump(le_machine, "le_machine.pkl")
joblib.dump(le_type, "le_type.pkl")
joblib.dump(scaler, "scaler.pkl")

print("💾 Multi-output model and encoders saved successfully!")
