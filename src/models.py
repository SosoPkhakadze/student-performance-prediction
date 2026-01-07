import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix

def load_data():
    if os.path.exists("data/processed/cleaned_student_data.csv"):
        return pd.read_csv("data/processed/cleaned_student_data.csv")
    else:
        raise FileNotFoundError("Run data_processing.py first!")

def train_and_evaluate():
    print("--- Starting Machine Learning Pipeline ---")
    df = load_data()
    
    # --- TASK 1: REGRESSION (Predicting Final Grade G3) ---
    print("\n1. Training Regression Models (Predicting Score)...")
    
    # Prepare X (Features) and y (Target)
    # Drop G3 (Target) and pass_fail (Classification Target)
    X = df.drop(['G3', 'pass_fail'], axis=1) 
    y_reg = df['G3']
    
    # Split Data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    # Model A: Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    
    # Model B: Decision Tree
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    y_pred_tree = tree_reg.predict(X_test)
    
    # Evaluation Metrics
    lin_r2 = r2_score(y_test, y_pred_lin)
    tree_r2 = r2_score(y_test, y_pred_tree)
    lin_mse = mean_squared_error(y_test, y_pred_lin)
    
    print(f"Linear Regression R2: {lin_r2:.4f}")
    print(f"Decision Tree R2: {tree_r2:.4f}")
    
    # --- TASK 2: CLASSIFICATION (Predicting Pass/Fail) - BONUS MODEL ---
    print("\n2. Training Classification Model (Predicting Pass/Fail)...")
    
    y_class = df['pass_fail']
    # Re-split to ensure matching indices (though random_state=42 makes it consistent)
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # Model C: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_c, y_train_c)
    y_pred_class = log_reg.predict(X_test_c)
    
    acc = accuracy_score(y_test_c, y_pred_class)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    
    # --- SAVE RESULTS TO REPORT ---
    os.makedirs("reports/results", exist_ok=True)
    with open("reports/results/model_performance.txt", "w") as f:
        f.write("=== MODEL PERFORMANCE REPORT ===\n\n")
        f.write("Problem Type: Regression (Predicting G3)\n")
        f.write(f"Model 1: Linear Regression\n - R2 Score: {lin_r2:.4f}\n - MSE: {lin_mse:.4f}\n\n")
        f.write(f"Model 2: Decision Tree Regressor\n - R2 Score: {tree_r2:.4f}\n\n")
        f.write("Comparison:\n")
        if lin_r2 > tree_r2:
            f.write("Linear Regression performed better. This suggests the relationships are mostly linear.\n")
        else:
            f.write("Decision Tree performed better. This suggests non-linear patterns.\n")
            
        f.write("\n----------------------------------\n")
        f.write("Problem Type: Classification (Pass/Fail)\n")
        f.write(f"Model 3: Logistic Regression\n - Accuracy: {acc:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test_c, y_pred_class)))
        
    print("\nResults saved to reports/results/model_performance.txt")

if __name__ == "__main__":
    train_and_evaluate()