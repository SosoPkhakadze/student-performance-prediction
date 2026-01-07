import sys
import os
import pandas as pd
import numpy as np
from src.data_processing import clean_and_process_data
from src.models import train_and_evaluate, load_data
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def clear_screen():
    # Simple clear screen command for Windows
    os.system('cls')

def print_header():
    print("=================================================")
    print("   🎓 STUDENT PERFORMANCE PREDICTION SYSTEM 🎓   ")
    print("=================================================")
    print("1. View Data Statistics")
    print("2. Run Model Training & Evaluation")
    print("3. Live Prediction Demo (Interactive)")
    print("4. Exit")
    print("=================================================")

def show_statistics():
    print("\n[LOADING DATA...]")
    # We need to reconstruct the dataframe using our processing script logic
    # or load the processed one
    if os.path.exists("data/processed/cleaned_student_data.csv"):
        df = pd.read_csv("data/processed/cleaned_student_data.csv")
        print(f"\nTotal Students: {df.shape[0]}")
        print(f"Average Final Grade: {df['G3'].mean():.2f} / 20")
        print("\nFeature Correlation with Grade (Top 5):")
        corr = df.select_dtypes(include=[np.number]).corr()['G3'].sort_values(ascending=False)
        print(corr.head(5))
    else:
        print("Error: Processed data not found. Run option 2 first.")
    input("\nPress Enter to return...")

def run_training():
    print("\n[TRAINING MODELS...]")
    # This runs the logic we wrote in src/models.py
    train_and_evaluate()
    input("\nPress Enter to return...")

def live_prediction():
    print("\n--- LIVE PREDICTION MODE ---")
    
    # Load data and retrain a simple linear model for the demo
    df = load_data()
    X = df.drop(['G3', 'pass_fail'], axis=1)
    y = df['G3']
    
    # Simple training for the demo session
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate averages for default values
    defaults = X.mean()
    
    print("Enter student details below:")
    try:
        g1 = float(input("1st Period Grade (0-20): "))
        g2 = float(input("2nd Period Grade (0-20): "))
        study = float(input("Weekly Study Time (1-4): "))
        failures = float(input("Past Failures (0-4): "))
        
        # Create input dataframe
        input_data = defaults.to_frame().T
        input_data['G1'] = g1
        input_data['G2'] = g2
        input_data['studytime'] = study
        input_data['failures'] = failures
        
        # Predict
        pred = model.predict(input_data)[0]
        
        print(f"\n>> PREDICTED FINAL GRADE: {pred:.2f} / 20")
        if pred >= 10:
            print(">> STATUS: PASS ✅")
        else:
            print(">> STATUS: AT RISK / FAIL ❌")
            
    except ValueError:
        print("Invalid input! Please enter numbers only.")
    
    input("\nPress Enter to return...")

def main():
    while True:
        clear_screen()
        print_header()
        choice = input("Select an option (1-4): ")
        
        if choice == '1':
            show_statistics()
        elif choice == '2':
            run_training()
        elif choice == '3':
            live_prediction()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()