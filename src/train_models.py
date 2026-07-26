import os
import sys
sys.path.insert(0, '.')
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

def run_pipeline():
    # Set styles
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", palette="muted")
    
    dataset_path = os.path.join('data', 'credit_card_approval_dataset.csv')
    if not os.path.exists(dataset_path):
        print("Dataset not found. Generating new dataset...")
        from data.generate_dataset import generate_credit_dataset
        df = generate_credit_dataset()
    else:
        df = pd.read_csv(dataset_path)
        
    print(f"Loaded dataset with shape: {df.shape}")

    # Output directory for plots & models
    eda_dir = os.path.join('static', 'images', 'eda')
    model_dir = 'model'
    os.makedirs(eda_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Exploratory Data Analysis (EDA) & Visualizations
    # -------------------------------------------------------------
    print("Generating EDA visualizations...")

    # Plot 1: Count Plots for Categorical & Target
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor('#0f172a')
    
    palette = ['#38bdf8', '#f43f5e', '#818cf8', '#34d399', '#fbbf24']
    
    sns.countplot(ax=axes[0, 0], data=df, x='Approval_Status', palette=['#ef4444', '#10b981'])
    axes[0, 0].set_title('Approval Status (0=Rejected, 1=Approved)', color='white', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticklabels(['Rejected', 'Approved'])

    sns.countplot(ax=axes[0, 1], data=df, x='Gender', palette=['#ec4899', '#3b82f6'])
    axes[0, 1].set_title('Gender Distribution', color='white', fontsize=12, fontweight='bold')

    sns.countplot(ax=axes[0, 2], data=df, x='Income_Type', palette='Set2')
    axes[0, 2].set_title('Income Type Distribution', color='white', fontsize=12, fontweight='bold')
    axes[0, 2].tick_params(axis='x', rotation=25)

    sns.countplot(ax=axes[1, 0], data=df, x='Housing_Type', palette='Set3')
    axes[1, 0].set_title('Housing Type Distribution', color='white', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=25)

    sns.countplot(ax=axes[1, 1], data=df, x='Education_Level', palette='Spectral')
    axes[1, 1].set_title('Education Level', color='white', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=25)

    sns.countplot(ax=axes[1, 2], data=df, x='Past_Due_Records', palette='Reds')
    axes[1, 2].set_title('Past-Due Records Count', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'count_plots.png'), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # Plot 2: Distribution Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0f172a')

    sns.histplot(ax=axes[0, 0], data=df, x='Annual_Income', hue='Approval_Status', kde=True, palette=['#ef4444', '#10b981'])
    axes[0, 0].set_title('Annual Income ($) Distribution by Approval', color='white', fontsize=12, fontweight='bold')

    sns.histplot(ax=axes[0, 1], data=df, x='Age', hue='Approval_Status', kde=True, palette=['#ef4444', '#10b981'])
    axes[0, 1].set_title('Age Distribution by Approval', color='white', fontsize=12, fontweight='bold')

    sns.histplot(ax=axes[1, 0], data=df, x='Employment_Duration', hue='Approval_Status', kde=True, palette=['#ef4444', '#10b981'])
    axes[1, 0].set_title('Employment Duration (Years) by Approval', color='white', fontsize=12, fontweight='bold')

    sns.histplot(ax=axes[1, 1], data=df, x='Credit_Inquiries', hue='Approval_Status', discrete=True, palette=['#ef4444', '#10b981'])
    axes[1, 1].set_title('Credit Inquiries (Last 6M) by Approval', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'distribution_plots.png'), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # -------------------------------------------------------------
    # 2. Data Preprocessing & Feature Engineering
    # -------------------------------------------------------------
    print("Preprocessing data and feature engineering...")
    
    df_model = df.copy()
    
    # Drop non-predictive ID column
    if 'Applicant_ID' in df_model.columns:
        df_model = df_model.drop(columns=['Applicant_ID'])
        
    # Feature Engineering: Create binary Past Due Risk label (Scenario 2)
    df_model['Past_Due_Risk_Binary'] = (df_model['Past_Due_Records'] > 0).astype(int)
    
    # Categorical columns encoding
    cat_cols = ['Gender', 'Marital_Status', 'Housing_Type', 'Education_Level', 'Income_Type']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    # Features and Target
    X = df_model.drop(columns=['Approval_Status'])
    y = df_model['Approval_Status']
    
    feature_names = X.columns.tolist()

    # Heatmap of correlation matrix
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor('#0f172a')
    corr = df_model.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix', color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'correlation_heatmap.png'), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------------
    # 3. Model Training & Evaluation (4 Algorithms)
    # -------------------------------------------------------------
    print("Training 4 Machine Learning Models...")
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    }

    if HAS_XGBOOST:
        models['XGBoost Classifier'] = XGBClassifier(n_estimators=100, learning_rate=0.08, random_state=42, eval_metric='logloss')
    else:
        models['XGBoost (Gradient Boosting)'] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, random_state=42)

    results = {}
    trained_model_objs = {}
    conf_matrices = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else acc
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        }
        trained_model_objs[name] = model
        conf_matrices[name] = cm
        print(f"[{name}] Accuracy: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")

    # Plot Model Comparison Bar Chart
    res_df = pd.DataFrame(results).T
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f172a')
    res_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)
    ax.set_title('Model Performance Metrics Comparison', color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score (0.0 to 1.0)', color='white')
    ax.set_ylim(0.5, 1.02)
    plt.xticks(rotation=0, color='white', fontsize=11)
    plt.yticks(color='white')
    plt.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'model_comparison.png'), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # Plot Confusion Matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('#0f172a')
    model_names = list(models.keys())
    
    for idx, name in enumerate(model_names):
        r, c = idx // 2, idx % 2
        sns.heatmap(conf_matrices[name], annot=True, fmt='d', cmap='Blues', ax=axes[r, c], cbar=False,
                    xticklabels=['Rejected', 'Approved'], yticklabels=['Rejected', 'Approved'])
        axes[r, c].set_title(f'{name} Confusion Matrix', color='white', fontsize=11, fontweight='bold')
        axes[r, c].set_ylabel('True Label', color='white')
        axes[r, c].set_xlabel('Predicted Label', color='white')

    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, 'confusion_matrices.png'), dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    # -------------------------------------------------------------
    # 4. Select & Serialize Best Model
    # -------------------------------------------------------------
    best_model_name = max(results, key=lambda k: results[k]['F1-Score'])
    best_model = trained_model_objs[best_model_name]
    print(f"\nBest Performing Model selected: {best_model_name} with F1-Score = {results[best_model_name]['F1-Score']}")

    # Extract Feature Importances if available
    feature_importances = {}
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importances = dict(zip(feature_names, [round(float(imp), 4) for imp in importances]))
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

    model_payload = {
        'best_model_name': best_model_name,
        'model': best_model,
        'all_models': trained_model_objs,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'feature_names': feature_names,
        'metrics': results,
        'feature_importances': feature_importances
    }

    model_file_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(model_payload, model_file_path)
    print(f"Saved best model artifact to {model_file_path}")

    return model_payload

if __name__ == '__main__':
    run_pipeline()
