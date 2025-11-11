import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("laundry_usage.csv")  

# Step 2: Convert time columns
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%H:%M", errors="coerce")
df["End_Time"] = df["Start_Time"] + pd.to_timedelta(df["Duration(min)"], unit="m")

# Step 3: Add useful features
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Day"].astype(str)

# Step 4: Encode categorical features
le_day = LabelEncoder()
le_machine = LabelEncoder()
le_type = LabelEncoder()

df["DayOfWeek_encoded"] = le_day.fit_transform(df["DayOfWeek"])
df["Machine_ID_encoded"] = le_machine.fit_transform(df["Machine_ID"])
df["MachineType_encoded"] = le_type.fit_transform(df["MachineType"])

# Step 5: Convert status to numeric (target variable)
df["Status"] = df["Status"].map({"Running": 1, "Idle": 0})

# Step 6: Select features and target
features = ["Machine_ID_encoded", "MachineType_encoded", "DayOfWeek_encoded", "Hour", "Duration(min)"]
target = "Status"

# Step 7: Scale numerical features
scaler = MinMaxScaler()
df[["Hour", "Duration(min)"]] = scaler.fit_transform(df[["Hour", "Duration(min)"]])

# Step 8: Split into training and testing sets
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data preprocessing complete!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("\nSample of cleaned data:")
print(df.head())
