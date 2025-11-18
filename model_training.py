import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_recall_fscore_support, roc_curve, auc)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("ENHANCED LAUNDRY MACHINE STATUS PREDICTION SYSTEM")
print("=" * 60)

# Load dataset
df = pd.read_csv("laundry_usage.csv")
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\nPerforming advanced feature engineering...")

# Time-based features
df["Start_Time"] = pd.to_datetime(df["Start_Time"], format="%H:%M", errors="coerce")
df["Hour"] = df["Start_Time"].dt.hour
df["DayOfWeek"] = df["Day"].astype(str)

# Cyclical encoding for time (preserves circular nature)
df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# Day-based features
df["Is_Weekend"] = df["DayOfWeek"].apply(lambda x: 1 if x in ["Sat", "Sun"] else 0)
df["High_Demand_Day"] = df["DayOfWeek"].apply(lambda x: 1 if x in ["Wed", "Sat", "Sun"] else 0)
df["Is_Weekday"] = 1 - df["Is_Weekend"]

# Time-of-day categories
def get_time_period(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 22:
        return "Evening"
    else:
        return "Night"

df["Time_Period"] = df["Hour"].apply(get_time_period)

# Peak hours (typical laundry times)
df["Is_Peak_Hour"] = df["Hour"].apply(lambda x: 1 if 9 <= x <= 11 or 18 <= x <= 21 else 0)

# Duration-based features
df["Is_Long_Cycle"] = (df["Duration(min)"] > df["Duration(min)"].median()).astype(int)
df["Duration_Category"] = pd.cut(df["Duration(min)"], bins=3, labels=["Short", "Medium", "Long"])

# Interaction features
df["Peak_Weekend"] = df["Is_Peak_Hour"] * df["Is_Weekend"]
df["Demand_Peak"] = df["High_Demand_Day"] * df["Is_Peak_Hour"]

# Label encoders
le_day = LabelEncoder()
le_machine = LabelEncoder()
le_type = LabelEncoder()
le_time_period = LabelEncoder()
le_duration_cat = LabelEncoder()

df["DayOfWeek_encoded"] = le_day.fit_transform(df["DayOfWeek"])
df["Machine_ID_encoded"] = le_machine.fit_transform(df["Machine_ID"])
df["MachineType_encoded"] = le_type.fit_transform(df["MachineType"])
df["Time_Period_encoded"] = le_time_period.fit_transform(df["Time_Period"])
df["Duration_Cat_encoded"] = le_duration_cat.fit_transform(df["Duration_Category"])

# Status encoding
df["Status"] = df["Status"].map({"Running": 1, "Idle": 0})

# Multi-output targets
df["Washer_Status"] = df.apply(lambda r: r["Status"] if r["MachineType"] == "Washer" else 0, axis=1)
df["Dryer_Status"] = df.apply(lambda r: r["Status"] if r["MachineType"] == "Dryer" else 0, axis=1)

# Enhanced feature set
features = [
    "Machine_ID_encoded", "MachineType_encoded", "DayOfWeek_encoded", 
    "Hour", "Hour_Sin", "Hour_Cos", "Duration(min)", 
    "High_Demand_Day", "Is_Weekend", "Is_Weekday", "Is_Peak_Hour",
    "Time_Period_encoded", "Is_Long_Cycle", "Duration_Cat_encoded",
    "Peak_Weekend", "Demand_Peak"
]

X = df[features].astype(float)
y = df[["Washer_Status", "Dryer_Status"]]

print(f"Feature engineering complete. Total features: {len(features)}")

# ============================================================================
# ADVANCED SCALING WITH STANDARDIZATION
# ============================================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# ============================================================================
# TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y["Washer_Status"]
)

print(f"\nData split: {len(X_train)} training, {len(X_test)} testing samples")

# ============================================================================
# MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================
print("\nTraining enhanced Random Forest model with optimization...")

# Base model with class balancing
base_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# Multi-output wrapper
multi_rf = MultiOutputClassifier(base_rf)
multi_rf.fit(X_train, y_train)

print("Model training complete.")

# Out-of-bag score (need to access the first estimator since it's wrapped in MultiOutputClassifier)
oob_score_washer = multi_rf.estimators_[0].oob_score_
oob_score_dryer = multi_rf.estimators_[1].oob_score_
print(f"Out-of-bag Score - Washer: {oob_score_washer:.4f}, Dryer: {oob_score_dryer:.4f}")

# ============================================================================
# CROSS-VALIDATION
# ============================================================================
print("\nPerforming 5-fold cross-validation...")
cv_scores_washer = cross_val_score(
    RandomForestClassifier(**base_rf.get_params()), 
    X_scaled, y["Washer_Status"], cv=5, scoring='accuracy'
)
cv_scores_dryer = cross_val_score(
    RandomForestClassifier(**base_rf.get_params()), 
    X_scaled, y["Dryer_Status"], cv=5, scoring='accuracy'
)

print(f"Washer CV Accuracy: {cv_scores_washer.mean():.4f} (+/- {cv_scores_washer.std():.4f})")
print(f"Dryer CV Accuracy: {cv_scores_dryer.mean():.4f} (+/- {cv_scores_dryer.std():.4f})")

# ============================================================================
# PREDICTIONS
# ============================================================================
y_pred = multi_rf.predict(X_test)
y_pred_proba = np.array([est.predict_proba(X_test)[:, 1] for est in multi_rf.estimators_]).T

# ============================================================================
# COMPREHENSIVE EVALUATION FUNCTION
# ============================================================================
def comprehensive_evaluation(y_test, y_pred, y_pred_proba, machine_name, output_idx):
    """Enhanced evaluation with multiple metrics and visualizations"""
    
    print(f"\n{'=' * 60}")
    print(f"{machine_name.upper()} PERFORMANCE METRICS")
    print(f"{'=' * 60}")
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Idle", "Running"]))
    
    # Create comprehensive visualization dashboard
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Confusion Matrix (Enhanced)
    ax1 = plt.subplot(3, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=["Idle", "Running"], 
                yticklabels=["Idle", "Running"],
                cbar_kws={'label': 'Count'})
    plt.title(f"{machine_name} Confusion Matrix\nAccuracy: {acc*100:.1f}%", 
              fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Status", fontweight='bold')
    plt.ylabel("Actual Status", fontweight='bold')
    
    # 2. Confusion Matrix (Normalized)
    ax2 = plt.subplot(3, 3, 2)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=["Idle", "Running"], 
                yticklabels=["Idle", "Running"])
    plt.title(f"{machine_name} Normalized Confusion Matrix", 
              fontsize=12, fontweight='bold')
    plt.xlabel("Predicted Status", fontweight='bold')
    plt.ylabel("Actual Status", fontweight='bold')
    
    # 3. Actual vs Predicted Distribution
    ax3 = plt.subplot(3, 3, 3)
    comparison_df = pd.DataFrame({
        'Actual': y_test.value_counts(),
        'Predicted': pd.Series(y_pred).value_counts()
    }).reindex([0, 1], fill_value=0)
    comparison_df.index = ['Idle', 'Running']
    comparison_df.plot(kind='bar', ax=ax3, color=['#3498db', '#e74c3c'], width=0.7)
    plt.title(f"{machine_name} Distribution Comparison", fontsize=12, fontweight='bold')
    plt.xlabel("Status", fontweight='bold')
    plt.ylabel("Count", fontweight='bold')
    plt.legend(title='Type', frameon=True)
    plt.xticks(rotation=0)
    
    # 4. ROC Curve
    ax4 = plt.subplot(3, 3, 4)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title(f'{machine_name} ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    # 5. Prediction Confidence Distribution
    ax5 = plt.subplot(3, 3, 5)
    plt.hist(y_pred_proba, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
    plt.xlabel('Prediction Probability', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.title(f'{machine_name} Prediction Confidence', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 6. Metrics Bar Chart
    ax6 = plt.subplot(3, 3, 6)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [acc, precision, recall, f1]
    colors_metric = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = plt.bar(metrics, values, color=colors_metric, alpha=0.8, edgecolor='black')
    plt.ylim([0, 1])
    plt.title(f'{machine_name} Performance Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 7. Error Analysis
    ax7 = plt.subplot(3, 3, 7)
    errors = (y_test != y_pred).astype(int)
    error_df = pd.DataFrame({
        'Correct': (1 - errors).sum(),
        'Incorrect': errors.sum()
    }, index=['Count'])
    error_df.T.plot(kind='pie', y='Count', ax=ax7, autopct='%1.1f%%',
                    colors=['#2ecc71', '#e74c3c'], startangle=90)
    plt.title(f'{machine_name} Prediction Accuracy', fontsize=12, fontweight='bold')
    plt.ylabel('')
    
    # 8. Class Distribution in Test Set
    ax8 = plt.subplot(3, 3, 8)
    class_dist = y_test.value_counts()
    colors_class = ['#f39c12', '#3498db']
    plt.pie(class_dist, labels=['Idle', 'Running'], autopct='%1.1f%%',
            colors=colors_class, startangle=90, explode=(0.05, 0.05))
    plt.title(f'{machine_name} Test Set Distribution', fontsize=12, fontweight='bold')
    
    # 9. Performance Summary Text
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    {machine_name.upper()} SUMMARY
    {'─' * 30}
    
    Accuracy:     {acc*100:.2f}%
    Precision:    {precision:.4f}
    Recall:       {recall:.4f}
    F1-Score:     {f1:.4f}
    ROC-AUC:      {roc_auc:.4f}
    
    Test Samples: {len(y_test)}
    Correct:       {(y_test == y_pred).sum()}
    Incorrect:     {(y_test != y_pred).sum()}
    
    Class Balance:
       Idle:    {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)
       Running: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, 
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle(f'{machine_name} Comprehensive Performance Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()
    
    return acc, precision, recall, f1, roc_auc


# ============================================================================
# EVALUATE BOTH MODELS
# ============================================================================
washer_metrics = comprehensive_evaluation(
    y_test["Washer_Status"], y_pred[:, 0], y_pred_proba[:, 0], "Washer", 0
)

dryer_metrics = comprehensive_evaluation(
    y_test["Dryer_Status"], y_pred[:, 1], y_pred_proba[:, 1], "Dryer", 1
)

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\nAnalyzing feature importance...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (name, ax) in enumerate(zip(["Washer", "Dryer"], axes)):
    importances = multi_rf.estimators_[idx].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot
    ax.barh(range(len(features)), importances[indices], color='teal', alpha=0.7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([features[i] for i in indices], fontsize=9)
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'{name} Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(importances[indices]):
        ax.text(v, i, f' {v:.3f}', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# Top 5 features for each
print("\nTop 5 Most Important Features:")
for idx, name in enumerate(["Washer", "Dryer"]):
    importances = multi_rf.estimators_[idx].feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    print(f"\n{name}:")
    for i, feat_idx in enumerate(top_indices, 1):
        print(f"  {i}. {features[feat_idx]}: {importances[feat_idx]:.4f}")

# ============================================================================
# COMPARATIVE MODEL ANALYSIS
# ============================================================================
print("\nCreating comparative analysis...")

fig, ax = plt.subplots(figsize=(12, 6))

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
washer_values = list(washer_metrics)
dryer_values = list(dryer_metrics)

x = np.arange(len(metrics_names))
width = 0.35

bars1 = ax.bar(x - width/2, washer_values, width, label='Washer', 
               color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, dryer_values, width, label='Dryer',
               color='#e74c3c', alpha=0.8, edgecolor='black')

ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Washer vs Dryer Model Performance Comparison', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11)
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# SAVE ENHANCED MODEL AND ARTIFACTS
# ============================================================================
print("\nSaving enhanced model and artifacts...")

# Save model
joblib.dump(multi_rf, "laundry_status_enhanced_model.pkl")

# Save preprocessing artifacts
artifacts = {
    'scaler': scaler,
    'label_encoders': {
        'day': le_day,
        'machine': le_machine,
        'type': le_type,
        'time_period': le_time_period,
        'duration_category': le_duration_cat
    },
    'feature_names': features,
    'model_metadata': {
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'n_features': len(features),
        'washer_metrics': {
            'accuracy': washer_metrics[0],
            'precision': washer_metrics[1],
            'recall': washer_metrics[2],
            'f1': washer_metrics[3],
            'roc_auc': washer_metrics[4]
        },
        'dryer_metrics': {
            'accuracy': dryer_metrics[0],
            'precision': dryer_metrics[1],
            'recall': dryer_metrics[2],
            'f1': dryer_metrics[3],
            'roc_auc': dryer_metrics[4]
        }
    }
}

joblib.dump(artifacts, "laundry_model_artifacts.pkl")

print("Model saved as: laundry_status_enhanced_model.pkl")
print("Artifacts saved as: laundry_model_artifacts.pkl")

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION COMPLETE")
print("=" * 60)
print(f"\nOverall Performance:")
print(f"   Washer Accuracy: {washer_metrics[0]*100:.2f}%")
print(f"   Dryer Accuracy:  {dryer_metrics[0]*100:.2f}%")
print(f"   Average:         {(washer_metrics[0] + dryer_metrics[0])/2*100:.2f}%")
print("\nReady for production deployment.")