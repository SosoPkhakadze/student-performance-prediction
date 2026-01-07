import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set the visual style
sns.set_style("whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

def load_data():
    """Loads the processed data."""
    # Ensure we look for the file relative to where the script is run
    if os.path.exists("data/processed/cleaned_student_data.csv"):
        return pd.read_csv("data/processed/cleaned_student_data.csv")
    else:
        raise FileNotFoundError("Processed data not found. Run data_processing.py first.")

def save_plot(fig, filename):
    """Saves the plot to the reports/figures directory."""
    os.makedirs("reports/figures", exist_ok=True)
    path = os.path.join("reports/figures", filename)
    fig.savefig(path, bbox_inches='tight')
    print(f"Generated: {path}")
    plt.close()

def run_eda():
    print("--- Starting Exploratory Data Analysis ---")
    df = load_data()

    # --- FIX: Reconstruct categorical columns for the Bonus Plot ---
    # The cleaning step converted these to numbers (0/1), so we convert them back for the plot labels
    if 'sex' not in df.columns and 'sex_M' in df.columns:
        df['sex'] = df['sex_M'].apply(lambda x: 'Male' if x == 1 else 'Female')
    
    if 'address' not in df.columns and 'address_U' in df.columns:
        df['address'] = df['address_U'].apply(lambda x: 'Urban' if x == 1 else 'Rural')
    # ---------------------------------------------------------------
    
    # 1. Distribution Plot (Target Variable G3)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['G3'], kde=True, bins=15, color='teal')
    plt.title('Distribution of Final Grades (G3)')
    plt.xlabel('Grade (0-20)')
    save_plot(plt, "01_distribution_G3.png")

    # 2. Correlation Heatmap (Numeric features only)
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    target_corr = corr['G3'].drop('G3')
    # Use features with correlation > 0.1 or < -0.1
    important_features = target_corr[abs(target_corr) > 0.1].index.tolist()
    important_features.append('G3')
    
    sns.heatmap(df[important_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap (Top Features)')
    save_plot(plt, "02_correlation_heatmap.png")

    # 3. Scatter Plot with Trend Line (G1 vs G3)
    plt.figure(figsize=(10, 6))
    sns.regplot(x='G1', y='G3', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    plt.title('Relationship: First Period Grade vs Final Grade')
    save_plot(plt, "03_scatter_G1_vs_G3.png")

    # 4. Box Plot (Study Time vs Grades)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='studytime', y='G3', data=df, palette='viridis', hue='studytime', legend=False)
    plt.title('Study Time vs Final Grades')
    plt.xlabel('Weekly Study Time (1: Low, 4: High)')
    save_plot(plt, "04_boxplot_studytime.png")

    # 5. Bar Plot (Failures vs Grades)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='failures', y='G3', data=df, palette='rocket', hue='failures', legend=False)
    plt.title('Impact of Past Failures on Final Grade')
    save_plot(plt, "05_barplot_failures.png")

    # 6. Advanced/Bonus Plot (FacetGrid)
    # Now this will work because we reconstructed 'sex' and 'address' at the top of this function
    g = sns.FacetGrid(df, col="sex", row="address", margin_titles=True, height=4)
    g.map(sns.histplot, "G3", kde=True, color="purple")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Grade Distribution by Demographics (Sex/Address)')
    save_plot(g, "06_bonus_demographics.png")

    # Print Statistics
    print("\n--- Key Statistics ---")
    print(f"Average Grade: {df['G3'].mean():.2f}")
    print(f"Std Deviation: {df['G3'].std():.2f}")
    print("\nTop Correlations with Grade:")
    print(df.select_dtypes(include=[np.number]).corr()['G3'].sort_values(ascending=False).head(5))

if __name__ == "__main__":
    run_eda()