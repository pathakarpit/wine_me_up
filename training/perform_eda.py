import pandas as pd
import numpy as np
import joblib
import os
import sys
from training.train_utils import DATA_FILE_PATH, MODEL_DIR, EDA_DIR, EDA_PATH
from scipy.stats import zscore
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor # Needed for Step 10

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def get_data_inspection_step(df_original):
    """ Step 1: Analyzes raw data. """
    print("   -> Processing Step 1: Data Inspection...")
    df = df_original.copy()
    
    if 'Id' in df.columns: df = df.drop('Id', axis=1)

    missing = df.isnull().sum()
    dup_count = df.duplicated().sum()
    df_clean = df.drop_duplicates()
    desc = df_clean.describe()
    
    code = """
# 1. Load & Inspect Data
df = pd.read_csv('WineQT.csv')
if 'Id' in df.columns: df = df.drop('Id', axis=1)

print(f"Duplicates: {df.duplicated().sum()}")
df = df.drop_duplicates()
"""
    comment = (
        "I first try to analyze the data and check for missing values and duplicates. "
        "Real chemical analysis rarely yields identical results for 11 float-point features, "
        "so exact duplicates are likely data entry errors or leakage."
    )

    return {
        'title': "1. Data Loading & Inspection",
        'type': 'dataset_overview', 
        'code': code,
        'comment': comment,
        'data': {
            'head': df_clean.head(), 
            'missing': missing, 
            'duplicates_removed': dup_count, 
            'description': desc,
            'shape': df_clean.shape   # <--- THIS WAS MISSING!
        },
        'cleaned_df': df_clean
    }

def get_target_distribution_step(df):
    """ Step 2: Target Distribution. """
    print("   -> Processing Step 2: Target Distribution...")
    
    counts = df['quality'].value_counts().sort_index()

    code = """
plt.figure(figsize=(8, 5))
ax = sns.countplot(x='quality', data=df, hue='quality', palette='viridis', legend=False)

# Add percentages
total = len(df)
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{100 * height / total:.1f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

plt.title("Distribution of Wine Quality")
plt.show()
"""
    comment = (
        "I started by analyzing the target distribution and found a severe class imbalance "
        "(over 80% are rated 5 or 6). This insight ruled out 'Accuracy' as a metric and led me to "
        "choose Stratified K-Fold splitting to ensure the model learns to identify rare, high-quality wines."
    )

    return {
        'title': "2. Target Distribution",
        'type': 'plot',
        'code': code,
        'comment': comment,
        'data': {'counts': counts}
    }

def get_correlation_heatmap_step(df):
    """ Step 3: Correlation Matrix. """
    print("   -> Processing Step 3: Correlation Matrix...")
    
    # 1. Calculate Correlation
    corr_matrix = df.corr()

    # 2. Code Snippet
    code = """
plt.figure(figsize=(12, 10))
# Calculate correlation matrix
corr = df.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Chemical Feature Correlation")
plt.show()
"""

    comment = (
        "**1. Feature Selection:** Alcohol and Sulphates are the strongest predictors of high quality.\n"
        "**2. Sanity Check:** Strong negative correlation between Acidity and pH (-0.7).\n"
        "**3. Redundancy:** Free Sulfur and Total Sulfur are highly correlated."
    )

    return {
        'title': "3. Correlation Heatmap",
        'type': 'heatmap',
        'code': code,
        'comment': comment,
        'data': {'corr_matrix': corr_matrix}
    }

def get_boxplot_analysis_step(df):
    """ Step 4: Boxplot Analysis. """
    print("   -> Processing Step 4: Boxplot Analysis...")

    features_to_check = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']
    
    code = """
features_to_check = ['alcohol', 'volatile acidity', 'sulphates', 'citric acid']

plt.figure(figsize=(14, 10))
for i, col in enumerate(features_to_check):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='quality', y=col, data=df, hue='quality', palette='coolwarm', legend=False)
    plt.title(f"{col} vs Quality")

plt.tight_layout()
plt.show()
"""
    comment = (
        "I didn't just rely on correlation coefficients. I visualized the distributions using boxplots "
        "to ensure the statistical signal was real. For example, I confirmed that **Alcohol** showed a "
        "distinct 'staircase' separation between quality classes, validating it as a primary predictor, "
        "whereas **Citric Acid** showed significant overlap, indicating it would be a weaker feature."
    )

    return {
        'title': "4. Feature Separation (Boxplots)",
        'type': 'boxplot_grid', 
        'code': code,
        'comment': comment,
        'data': {'features': features_to_check}
    }

def get_feature_engineering_step(df):
    """ Step 5: Feature Engineering Impact. """
    print("   -> Processing Step 5: Feature Engineering...")
    
    df_eng = df.copy()
    
    # 1. Engineering Logic
    df_eng['total_acidity'] = df_eng['fixed acidity'] + df_eng['volatile acidity'] + df_eng['citric acid']
    df_eng['acidity_to_density_ratio'] = df_eng['total_acidity'] / df_eng['density']
    df_eng['alcohol_density_ratio'] = df_eng['alcohol'] / df_eng['density']
    df_eng['sugar_alcohol_ratio'] = df_eng['residual sugar'] / df_eng['alcohol']
    
    # Calculate Correlations
    new_features = ['quality', 'total_acidity', 'alcohol_density_ratio', 'sugar_alcohol_ratio']
    correlations = df_eng[new_features].corr()['quality'].sort_values(ascending=False)

    code = """
def engineer_features(dataframe):
    df_eng = dataframe.copy()
    
    # 1. Total Acidity (Sum of all acids)
    df_eng['total_acidity'] = df_eng['fixed acidity'] + df_eng['volatile acidity'] + df_eng['citric acid']
    
    # 2. Acidity to Density Ratio
    df_eng['acidity_to_density_ratio'] = df_eng['total_acidity'] / df_eng['density']
    
    # 3. Alcohol to Density Ratio (Alcohol is lighter than water)
    df_eng['alcohol_density_ratio'] = df_eng['alcohol'] / df_eng['density']
    
    # 4. Sugar to Alcohol Ratio (Fermentation efficiency)
    df_eng['sugar_alcohol_ratio'] = df_eng['residual sugar'] / df_eng['alcohol']
    
    return df_eng

df_engineered = engineer_features(df)
print(df_engineered[['quality', 'total_acidity', 'alcohol_density_ratio']].corr()['quality'])
"""
    
    comment = (
        "I experimented with feature engineering, specifically an **Alcohol-to-Density Ratio**. "
        "Interestingly, I found that this complex feature had the **same correlation (0.48) as raw Alcohol**. "
        "This suggests that Alcohol is the dominant driver and Density acts mostly as noise in this relationship. "
        "I chose to keep the raw feature for linear models to preserve interpretability, but I retained the ratio "
        "for Tree-based ensembles to capture potential non-linear edge cases."
    )

    return {
        'title': "5. Feature Engineering Experiments",
        'type': 'correlation_comparison', 
        'code': code,
        'comment': comment,
        'data': {'new_correlations': correlations}
    }

def get_feature_distribution_step(df):
    """ Step 6: All Feature Distributions (KDE). """
    print("   -> Processing Step 6: Feature Distributions (KDE)...")
    
    features = [col for col in df.columns if col != 'quality']
    
    code = """
import math
features = [col for col in df.columns if col != 'quality']
n_cols = 1
n_rows = math.ceil(len(features) / n_cols)

plt.figure(figsize=(15, 4 * n_rows))
for i, col in enumerate(features):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # KDE Plot with common_norm=False (Crucial for class imbalance)
    sns.kdeplot(data=df, x=col, hue='quality', palette='viridis', 
                common_norm=False, fill=True, alpha=0.3)
    
    plt.title(f'{col} Distribution by Quality')
    plt.xlabel(col)

plt.tight_layout()
plt.show()
"""
    comment = (
        "I used **KDE plots** with `common_norm=False` to handle the class imbalance visually. "
        "This allowed me to compare the distribution *shapes* of rare classes (like Quality 8) against common ones. "
        "It visually confirmed that **Alcohol** and **Volatile Acidity** offered the distinct distributional separation "
        "needed for a model to learn effectively, whereas features like **pH** showed almost complete overlap."
    )

    return {
        'title': "6. Detailed Feature Distributions",
        'type': 'kde_grid', 
        'code': code,
        'comment': comment,
        'data': {'features': features}
    }

def get_outlier_analysis_step(df):
    """ Step 7: Z-Score Outlier Analysis. """
    print("   -> Processing Step 7: Outlier Analysis...")
    
    features = [col for col in df.columns if col != 'quality']
    
    # Calculate Z-scores
    z_scores = df[features].apply(zscore)
    
    # Count outliers > 3 std dev
    outliers = (np.abs(z_scores) > 3).sum()
    
    code = """
from scipy.stats import zscore

features = [col for col in df.columns if col != 'quality']
z_scores = df[features].apply(zscore)

# Count rows that are outliers (more than 3 std devs away)
outliers = (np.abs(z_scores) > 3).sum()

plt.figure(figsize=(12, 6))
sns.barplot(x=outliers.index, y=outliers.values, hue=outliers.index, palette='Reds_r', legend=False)

plt.title("Count of Extreme Outliers (>3 Std Dev) per Feature")
plt.ylabel("Number of Wines")
plt.xticks(rotation=45)
plt.show()
"""

    comment = (
        "Because of these outliers, I cannot use **MinMaxScaler** (which squishes everything based on the max value). "
        "I must use **RobustScaler** or **StandardScaler** to prevent these few extreme wines from ruining the scale for everyone else.\n\n"
        "Since **linear regression** is sensitive to outliers (it tries to draw a line to reach them), "
        "this reinforces my choice to use **Tree-based models (Random Forest/XGBoost)** later, "
        "as they naturally handle outliers by simply splitting them into their own leaf nodes.\n\n"
        "I calculated **Z-scores** to identify distributional anomalies. Finding that **Residual Sugar** and **Chlorides** "
        "were highly skewed, I decided against simple min-max scaling, which would be distorted by these outliers."
    )

    return {
        'title': "7. Outlier Detection (Z-Scores)",
        'type': 'outlier_plot', 
        'code': code,
        'comment': comment,
        'data': {'outliers': outliers}
    }

def get_feature_profile_step(df):
    """ Step 8: Chemical Profile (Normalized Means). """
    print("   -> Processing Step 8: Chemical Profile Analysis...")

    means = df.groupby('quality').mean()
    means_normalized = (means - means.min()) / (means.max() - means.min())

    code = """
# Group by quality and get the mean of every feature
means = df.groupby('quality').mean()

# Normalize data for this plot (Min-Max)
means_normalized = (means - means.min()) / (means.max() - means.min())

plt.figure(figsize=(15, 8))
sns.heatmap(means_normalized, annot=True, cmap='Blues', linewidths=0.5)
plt.title("Average Feature Value by Quality (Normalized 0-1)")
plt.show()
"""

    comment = (
        "I normalized and aggregated the feature means to visualize the chemical 'profile' of each quality tier. "
        "This plot highlighted clear **linear progressions** in features like **Alcohol** and **Volatile Acidity**, "
        "justifying the use of linear baseline models.\n\n"
        "However, it also exposed the **non-linear behavior** of secondary features like **pH**, "
        "which confirmed the need for **tree-based ensemble methods** for the final solution."
    )

    return {
        'title': "8. Chemical Profile Analysis",
        'type': 'profile_heatmap', 
        'code': code,
        'comment': comment,
        'data': {'means_normalized': means_normalized}
    }

def get_interactive_plot_step(df):
    """ Step 9: Interactive Plotly Animation. """
    print("   -> Processing Step 9: Interactive Plot (Sulfur Evolution)...")
    
    df_plot = df.copy()
    df_plot = df_plot.sort_values('quality')
    df_plot['quality'] = df_plot['quality'].astype(str)
    
    fig = px.scatter(
        df_plot, 
        x="free sulfur dioxide", 
        y="total sulfur dioxide", 
        animation_frame="quality", 
        color="alcohol", 
        size="sulphates", 
        hover_data=['pH', 'density'], 
        range_x=[0, df_plot['free sulfur dioxide'].max()],
        range_y=[0, df_plot['total sulfur dioxide'].max()],
        title="Evolution of Sulphur Dioxide Levels across Quality Scores"
    )

    code = """
import plotly.express as px

# 1. Sort the dataframe by quality (Animations need sorted frames)
df = df.sort_values('quality')

# 2. Ensure quality is a string for the slider
df['quality'] = df['quality'].astype(str)

fig = px.scatter(
    df, 
    x="free sulfur dioxide", 
    y="total sulfur dioxide", 
    animation_frame="quality",      # The magic slider!
    color="alcohol",                # Color dots by alcohol content
    size="sulphates",               # Size dots by sulphates
    hover_data=['pH', 'density'],   # Show extra info on hover
    range_x=[0, df['free sulfur dioxide'].max()],
    range_y=[0, df['total sulfur dioxide'].max()],
    title="Evolution of Sulphur Dioxide Levels across Quality Scores"
)

fig.show()
"""

    comment = (
        "In analyzing the relationship between Free and Total Sulfur Dioxide, I expected a strict linear correlation. "
        "However, the interactive plot revealed a **significant horizontal spread** rather than a perfect diagonal line. "
        "This variance indicates that the amount of **'Bound Sulfur'** (Total minus Free) varies drastically between wines.\n\n"
        "This insight was critical because it proved that we cannot simply drop one of these features due to multicollinearity. "
        "The 'Bound Sulfur' variance contains unique chemical information about the wine's preservation state, "
        "validating my decision to retain both features."
    )

    return {
        'title': "9. Interactive Analysis (Sulfur Evolution)",
        'type': 'plotly', 
        'code': code,
        'comment': comment,
        'data': {'fig': fig}
    }

def get_feature_importance_step(df):
    """ Step 10: Random Forest Feature Importance. """
    print("   -> Processing Step 10: Feature Importance (RF)...")

    # 1. Setup Data
    X = df.drop('quality', axis=1)
    y = df['quality']

    # 2. Train Quick Model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # 3. Get Importance
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    code = """
from sklearn.ensemble import RandomForestRegressor

# Setup data
X = df.drop('quality', axis=1)
y = df['quality']

# Train quick model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get importance
importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importances, hue='Feature', palette='viridis', legend=False)

plt.title("Random Forest Feature Importance (What drives Quality?)")
plt.show()
"""

    comment = (
        "I validated my feature selection using a **Random Forest Regressor** to extract 'Feature Importance.' "
        "This confirmed that **Alcohol** was the primary driver of quality, consistent with my EDA findings.\n\n"
        "However, unlike the linear correlation matrix, the Random Forest revealed that **Sulphates** played a "
        "larger role in **non-linear decision boundaries**, prompting me to retain it despite its moderate linear correlation."
    )

    return {
        'title': "10. Feature Importance Verification",
        'type': 'feature_importance_plot', # Frontend uses this
        'code': code,
        'comment': comment,
        'data': {'importances': importances}
    }

# ==========================================
# 2. MAIN ORCHESTRATOR
# ==========================================

def generate_eda_artifacts():
    print("--- Starting EDA Generation ---")
    
    try:
        raw_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        sys.exit(f"Error: Data file not found at {DATA_FILE_PATH}")

    artifacts = {'eda_sections': []}

    # --- EXECUTE STEPS ---
    
    # Step 1: Inspection
    step1 = get_data_inspection_step(raw_df)
    artifacts['eda_sections'].append(step1)
    current_df = step1['cleaned_df']
    artifacts['full_data'] = current_df 

    # Step 2: Target Dist
    step2 = get_target_distribution_step(current_df)
    artifacts['eda_sections'].append(step2)
    
    # Step 3: Heatmap
    step3 = get_correlation_heatmap_step(current_df)
    artifacts['eda_sections'].append(step3)

    # Step 4: Boxplots
    step4 = get_boxplot_analysis_step(current_df)
    artifacts['eda_sections'].append(step4)
    
    # Step 5: Engineering
    step5 = get_feature_engineering_step(current_df)
    artifacts['eda_sections'].append(step5)
    
    # Step 6: Feature Dist (KDE)
    step6 = get_feature_distribution_step(current_df)
    artifacts['eda_sections'].append(step6)
    
    # Step 7: Outliers
    step7 = get_outlier_analysis_step(current_df)
    artifacts['eda_sections'].append(step7)
    
    # Step 8: Chemical Profile
    step8 = get_feature_profile_step(current_df)
    artifacts['eda_sections'].append(step8)
    
    # Step 9: Interactive Plotly
    step9 = get_interactive_plot_step(current_df)
    artifacts['eda_sections'].append(step9)

    # Step 10: Feature Importance (NEW)
    step10 = get_feature_importance_step(current_df)
    artifacts['eda_sections'].append(step10)

    # --- SAVE ---
    eda_dir = EDA_DIR
    if not os.path.exists(eda_dir): os.makedirs(eda_dir)
        
    save_path = EDA_PATH
    joblib.dump(artifacts, save_path)
    print(f"\n✅ EDA Artifacts saved to: {save_path}")

if __name__ == "__main__":
    generate_eda_artifacts()