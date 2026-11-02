import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 1. Reproducible Synthetic Dataset Generation (N=1000)
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'Age': np.random.normal(loc=38.5, scale=12.0, size=n_samples).astype(int),
    'Annual_Income': np.random.lognormal(mean=11.0, sigma=0.4, size=n_samples),
    'Spending_Score': np.random.randint(1, 101, size=n_samples),
    'Customer_Segment': np.random.choice(['Basic', 'Premium', 'Enterprise'], size=n_samples, p=[0.52, 0.33, 0.15]),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.81, 0.19])
})

# Introduce 4.5% Missing Values in Annual_Income
missing_idx = np.random.choice(data.index, size=45, replace=False)
data.loc[missing_idx, 'Annual_Income'] = np.nan

# 2. Data Hygiene & Missing Value Imputation
income_median = data['Annual_Income'].median()
data['Annual_Income_Imputed'] = data['Annual_Income'].fillna(income_median)

# 3. Outlier Treatment (IQR Winsorization)
q1 = data['Annual_Income_Imputed'].quantile(0.25)
q3 = data['Annual_Income_Imputed'].quantile(0.75)
iqr = q3 - q1
upper_bound = q3 + 1.5 * iqr
data['Annual_Income_Capped'] = np.where(
    data['Annual_Income_Imputed'] > upper_bound, 
    upper_bound, 
    data['Annual_Income_Imputed']
)

# A. Visualization 1: Histogram + KDE (Matplotlib / Seaborn) -> Saves as PNG
plt.figure(figsize=(9, 5))
sns.histplot(data['Annual_Income_Capped'], kde=True, color='#2b5c8f', bins=20)
plt.axvline(income_median, color='#d9534f', linestyle='--', label=f'Median: ${income_median:,.2f}')
plt.title('Figure 1: Distribution of Annual Income (Post-Imputation & Capping)', fontsize=12, fontweight='bold')
plt.xlabel('Annual Income ($)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig('annual_income_distribution.png', dpi=300)
plt.show()
plt.close()
print("Saved: annual_income_distribution.png")

# B. Visualization 2: Feature Correlation Heatmap -> Saves as PNG
plt.figure(figsize=(8, 6))
numeric_df = data[['Age', 'Annual_Income_Capped', 'Spending_Score', 'Churn']]
corr_matrix = numeric_df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
plt.title('Figure 2: Pearson Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()
plt.close()
print("Saved: correlation_heatmap.png")

# C. Interactive Visualization: Plotly Express -> Saves as HTML Dashboard
fig = px.scatter(
    data,
    x='Annual_Income_Capped',
    y='Spending_Score',
    color='Customer_Segment',
    symbol='Churn',
    title='Figure 3: Interactive Segment Exploration (Income vs. Spending Score)',
    labels={'Annual_Income_Capped': 'Annual Income ($)', 'Spending_Score': 'Spending Score (1-100)'},
    template='plotly_white'
)
fig.write_html('interactive_segmentation.html')
print("Saved: interactive_segmentation.html")
