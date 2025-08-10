 Bank Attrition Analysis
This project analyzes customer data from a retail bank to identify patterns and factors influencing customer attrition (churn). The goal is to explore demographic, behavioral, and financial attributes to better understand why customers leave and provide insights for retention strategies.

Key Objectives:

Perform data cleaning & preprocessing (handling missing values, standardizing columns, managing duplicates).

Explore customer demographics (age, gender, state) and their correlation with churn.

Analyze financial factors like credit score, account balance, salary category, and debt-to-income ratio.

Investigate product usage behavior such as number of products, loan holding, card type, and loyalty score.

Use visualizations to highlight churn trends and high-risk customer segments.

Key Deliverables:

Cleaned and structured dataset ready for modeling.

Exploratory Data Analysis (EDA) charts showing churn drivers.

Insights for potential customer retention strategies.

Tools & Technologies:
Python (Pandas, NumPy, Matplotlib, Seaborn) | Jupyter Notebook


# Bank-Attrition-Analyis
This project analyzes customer data from a retail bank to identify patterns and factors influencing customer attrition (churn). The goal is to explore demographic, behavioral, and financial attributes to better understand why customers leave and provide insights for retention strategies.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset, skipping the metadata row
df = pd.read_csv('https://gitlab.crio.do/me_notebook/me_jupyter_bankattritionanalysis/-/raw/master/bank_attrition_dataset.csv',header = 0, skiprows=[1])

# Display the first few rows
df.head()

# Print the column names
# print(df.columns)

# Standardize the column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Print the standardized column names to verify
df.columns

# Check missing values
# print(df.isnull().sum())

# Salary → replace with median
df['salary'] = df['salary'].fillna(df['salary'].median())

# Balance → replace with 0
df['balance'] = df['balance'].fillna(0)

# Satisfaction Score → median
df['satisfaction_score'] = df['satisfaction_score'].fillna(df['satisfaction_score'].median())

# Gender → drop rows with missing gender
df = df.dropna(subset=['gender'])

# Card Type → fill with most frequent value
df['card_type'] = df['card_type'].fillna(df['card_type'].mode()[0])

# Check again
print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Standardize card_type: uppercase & remove spaces
df['card_type'] = df['card_type'].str.lower().str.replace(' ', '_')

# Identify numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print("Numeric Columns:", numeric_cols)

# Count unique values for each numeric column
unique_counts = df[numeric_cols].nunique()
print("\nUnique Counts:\n", unique_counts)

# Boxplots for Salary and Balance
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.boxplot(df['salary'])
plt.title("Salary - Before Outlier Treatment")

plt.subplot(1, 2, 2)
plt.boxplot(df['balance'])
plt.title("Balance")
plt.show()

# IQR method for Salary
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count outliers
outliers = df[(df['salary'] < lower_bound) | (df['salary'] > upper_bound)]
print("Number of Salary Outliers:", len(outliers))

# Cap outliers to upper bound
df['salary'] = np.where(df['salary'] > upper_bound, upper_bound, df['salary'])

# Plot after handling
plt.boxplot(df['salary'])
plt.title("Salary - After Outlier Treatment")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/12f5b275-a3fc-409d-8057-03fc98f34dff" />

# Step 1: Basic Summary Statistics
salary_mean = df['salary'].mean()
salary_median = df['salary'].median()

balance_mean = df['balance'].mean()
balance_median = df['balance'].median()

credit_mean = df['creditscore'].mean()
credit_median = df['creditscore'].median()

print(f"salary - Mean: {salary_mean:.2f}, Median: {salary_median}")
print(f"balance - Mean: {balance_mean:.2f}, Median: {balance_median}")
print(f"creditscore - Mean: {credit_mean:.2f}, Median: {credit_median}")

# Step 2: Count categories
categorical_vars = ['gender', 'card_type', 'hasloan', 'hasfd']

for col in categorical_vars:
    print(f"\nValue Counts for {col}:\n", df[col].value_counts())

# Optional: Visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, col in enumerate(categorical_vars):
    sns.countplot(x=df[col], ax=axes[i], palette="viridis")
    axes[i].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c0908a00-9ece-47cf-a649-ecf177ab7105" />

# Step 3: Boxplot for Balance
plt.figure(figsize=(6, 4))
sns.boxplot(x=df['balance'], color='skyblue')
plt.title("Balance Distribution")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/c743eeb3-9105-46fb-81a5-282dcea517fd" />

# Step 4: Scatter plot - sample of 200
sample_df = df.sample(n=200, random_state=42)

plt.figure(figsize=(6, 4))
sns.scatterplot(x='creditscore', y='balance', data=sample_df, hue='gender')
plt.title("Credit Score vs Balance")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e9aad074-8542-41ef-8f8b-1cc0d0885cc3" />
# Step 5: New features

# Debt-to-Income Ratio
df['debt_to_income_ratio'] = (df['balance'] + (df['hasloan'] * df['salary'] * 0.3)) / df['salary']

# Loyalty Score
df['loyalty_score'] = (df['tenure'] * df['satisfaction_score']) / (1 + df['numcomplain'])

df.head()

# Step 6: Complaints vs State Average
state_avg_complaints = df.groupby('state')['numcomplain'].transform('mean')
df['high_complainer'] = (df['numcomplain'] > state_avg_complaints).astype(int)

df[['state', 'numcomplain', 'high_complainer']].head()
df.isnull().sum()

# Step 7: Salary Category
conditions = [
    (df['salary'] <= 50000),
    (df['salary'] > 50000) & (df['salary'] <= 100000),
    (df['salary'] > 100000) & (df['salary'] <= 150000),
    (df['salary'] > 150000) & (df['salary'] <= 200000),
    (df['salary'] > 200000)
]

bins = [0, 50000, 100000, 150000, 200000, np.inf]
labels = ['Low', 'Medium', 'High', 'Very High', 'Above 2 Lakhs']

df['salary_category'] = pd.cut(df['salary'], bins=bins, labels=labels)

# Count and plot
salary_cat_counts = df['salary_category'].value_counts()

plt.figure(figsize=(6, 4))
plt.bar(salary_cat_counts.index, salary_cat_counts.values, color='teal')
plt.title("Customer Distribution by Salary Category")
plt.xlabel("Salary Category")
plt.ylabel("Number of Customers")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e0543679-16ee-49e8-9e53-2a9363da1c7d" />


import matplotlib.pyplot as plt

# ✅ Average Number of Products Based on Customer Tenure
avg_products_by_tenure = df.groupby('tenure')['numofproducts'].mean()

# ✅ Group data by churn status to analyze salary and product usage
avg_stats_by_churn = df.groupby('exited').agg({
    'salary': 'mean',
    'numofproducts': 'mean'
})

# --- Plot: Average Salary - Stayed vs. Exited ---
plt.figure(figsize=(6,4))
avg_stats_by_churn['salary'].plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Average Salary - Stayed vs. Exited')
plt.xlabel('Churn Status (0 = Stayed, 1 = Exited)')
plt.ylabel('Average Salary')
plt.xticks(rotation=0)
plt.show()

# --- Plot: Average Number of Products - Stayed vs. Exited ---
plt.figure(figsize=(6,4))
avg_stats_by_churn['numofproducts'].plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Average Number of Products - Stayed vs. Exited')
plt.xlabel('Churn Status (0 = Stayed, 1 = Exited)')
plt.ylabel('Average Number of Products')
plt.xticks(rotation=0)
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7f106b34-50a6-4faf-b1f9-1ee5a006215b" />

# Step 7 churn_counts 
churn_counts = df['exited'].value_counts()
plt.pie(
    churn_counts,
    labels=['Stayed', 'Exited'],
    autopct="%.1f%%",
    startangle=90,
    colors=['skyblue', 'salmon']
)
plt.title("Customer Churn Proportion")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ae929eae-e74a-4a17-a4d1-e72733ab9c5a" />

# Step 8 state_churn 
state_churn = df.groupby('state')['exited'].sum()
plt.pie(
    state_churn,
    labels=state_churn.index,
    autopct="%.1f%%",
    startangle=90
)
plt.title("Churn Proportion by State")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6dcb7203-4588-422f-8db5-0e3773cfb6cc" />

sns.scatterplot(
    data=df,
    x="age",
    y="exited",
    hue="exited",
    palette={0: 'blue', 1: 'red'},
    alpha=0.6
)
plt.title("Scatter Plot: Age vs Number of Customers (Churned vs Non-Churned)")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/14285018-169a-4a44-989c-4c56c4b78072" />

sns.boxplot(x='exited', y='age', data=df, palette=['blue', 'red'])
plt.title("Box Plot: Age Distribution by Churn Status")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b1e4216a-5ec5-4450-a0d0-221f32b1faa8" />

gender_churn = df.groupby('gender')['exited'].mean().reset_index()
sns.barplot(x='gender', y='exited', data=gender_churn, palette='viridis')
plt.ylabel("Churn Rate")
plt.title("Churn Rate Distribution by Gender")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d539271d-0875-4a99-8312-766f2823fa18" />

state_gender_churn = df.groupby(['state', 'gender'])['exited'].mean().reset_index()
sns.barplot(x='state', y='exited', hue='gender', data=state_gender_churn)
plt.ylabel("Churn Rate")
plt.title("Churn Rate by State and Gender")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b03549fd-ca40-42f5-b2f4-bfc3ca8358e9" />

bins = [0, 50000, 100000, 150000, 200000, df['salary'].max()]
labels = ['<50K', '50-100K', '100-150K', '150-200K', '200K+']
df['income_group'] = pd.cut(df['salary'], bins=bins, labels=labels)

income_churn = df.groupby('income_group')['exited'].mean().reset_index()
sns.barplot(x='income_group', y='exited', data=income_churn, palette='coolwarm')
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Income Group")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2ae9eea5-66ad-42ed-981c-fe382aaa0305" />

region_gender_churn = df.groupby(['state', 'gender'])['exited'].mean().reset_index()

sns.barplot(x='state', y='exited', hue='gender', data=region_gender_churn)
plt.ylabel("Churn Rate")
plt.title("Churn Rate by Region and Gender")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7a80e783-b740-4a96-89d2-b856b8a399b2" />

bins = [0, 30, 40, 50, 60, df['age'].max()]
labels = ['<30', '30-40', '40-50', '50-60', '60+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

age_satisfaction = df[df['exited'] == 1].groupby('age_group')['satisfaction_score'].mean().reset_index()
sns.barplot(x='age_group', y='satisfaction_score', data=age_satisfaction, palette='mako')
plt.title("Churned Customer Satisfaction by Age Group")
plt.show()
print(df.columns)


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7e80f5eb-45ab-43ed-a834-7c3377529e5d" />


sns.countplot(x="income_group", hue="exited", data=df, palette="pastel")
plt.title("Income Group vs Exited Customers")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/0f46a997-ab21-4ff3-a0f5-8c05e9293bf4" />

state_churn_rate = df.groupby('state')['exited'].mean().reset_index()
fig = px.bar(
    state_churn_rate,
    x='state',
    y='exited',
    title="Churn Rate by State",
    labels={'exited': 'Churn Rate'}
)
fig.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/5a24060e-37b8-415d-85f1-d5d98c7bc2d5" />

#  Step 9: Credit Card Distribution Among Churned Customers
churned_customers = df[df['exited'] == 1]

plt.figure(figsize=(6,4))
sns.countplot(data=churned_customers, x='hascrcard', palette="coolwarm")
plt.title("Credit Card Ownership Distribution Among Exited Customers")
plt.xlabel("Has Credit Card (0 = No, 1 = Yes)")
plt.ylabel("Number of Churned Customers")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e373f5da-afdb-48a8-856c-4aea4c6ef66a" />

# Step 10: Churn Percentage by Credit Card Ownership
churn_group = df.groupby(['hascrcard', 'exited']).size().reset_index(name='count')
total_customers = df.groupby('hascrcard')['customerid'].count().reset_index(name='total')
churn_percent = churn_group.merge(total_customers, on='hascrcard')
churn_percent['percentage'] = (churn_percent['count'] / churn_percent['total']) * 100
plt.figure(figsize=(6,4))
sns.barplot(data=churn_percent, x='hascrcard', y='percentage', hue='exited')
plt.title("Global Customer Churn Percentage by Credit Card Status")
plt.xlabel("Has Credit Card (0 = No, 1 = Yes)")
plt.ylabel("Percentage of Total Customers")
plt.legend(title="Churned (Exited)")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/9ba86e08-88b1-40c9-ac43-ae889ae26324" />

#Step 11: Credit Card Type Distribution Across Churn Status
card_group = df.groupby(['card_type', 'exited']).size().reset_index(name='count')
total_card = df.groupby('card_type')['customerid'].count().reset_index(name='total')
card_group = card_group.merge(total_card, on='card_type')
card_group['percentage'] = (card_group['count'] / card_group['total']) * 100

plt.figure(figsize=(8,5))
sns.barplot(data=card_group, x='card_type', y='percentage', hue='exited')
plt.title("Credit Card Type Percentage Across Churn Status")
plt.xlabel("Credit Card Type")
plt.ylabel("Percentage of Total Customers")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/bc9e8aa1-6f9a-4883-aaef-c4d69b134b48" />

# Step 12: Loan Ownership vs Churn
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='hasloan', hue='exited')
plt.title("HasLoan Ownership Distribution by Churned Status")
plt.xlabel("Has Loan (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/098f9d6e-0033-4a0e-86c2-6f0c565410f9" />

# Step 13: Number of Products vs Churn (Churned customers only)
plt.figure(figsize=(6,4))
sns.countplot(data=churned_customers, x='card_type')
plt.title("Credit Card Type Distribution")
plt.xlabel("Card Type")
plt.ylabel("Number of Churned Customers")
plt.show()

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/86ccf753-5063-4b11-b18f-143bc1d1e662" />

# Step 14: Product Distribution Across Churn
product_counts = pd.DataFrame({
    'Has Credit Card': df.groupby('exited')['hascrcard'].sum(),
    'Has Loan': df.groupby('exited')['hasloan'].sum(),
    'Has FD': df.groupby('exited')['hasfd'].sum()
}).reset_index()

product_counts_melted = product_counts.melt(id_vars='exited', var_name='Product', value_name='Count')

plt.figure(figsize=(8,5))
ax = sns.barplot(data=product_counts_melted, x='Product', y='Count', hue='exited')
plt.title("Product Distribution Across Churn Status")
plt.xlabel("Product")
plt.ylabel("Number of Customers")
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/59141b48-cce1-48d2-988e-c9f829b4d25b" />

# Step 15: Average Product Usage by Tenure Groups Across Churn Status
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,5,10,15,20], labels=['0-5','6-10','11-15','16-20'])
tenure_avg = df.groupby(['tenure_group', 'exited'])['numofproducts'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=tenure_avg, x='tenure_group', y='numofproducts', hue='exited')
plt.title("Tenure Group vs Avg Number of Products by Churn Status")
plt.xlabel("Tenure Group (Years)")
plt.ylabel("Average Number of Products")
plt.legend(title="Exited", labels=['No', 'Yes'])
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/dbf99b7e-fcb4-4564-a31a-82659aefdd91" />

#Step 16: Average Number of Products by Credit Score Ranges Across Churn Status
df['creditscore_range'] = pd.cut(df['creditscore'], bins=[300,500,650,800,900], 
                                 labels=['300-500','501-650','651-800','801-900'])
credit_avg = df.groupby(['creditscore_range', 'exited'])['numofproducts'].mean().reset_index()

plt.figure(figsize=(8,5))
ax = sns.barplot(data=credit_avg, x='creditscore_range', y='numofproducts', hue='exited')
plt.title("Average Product Usage by Credit Score Range and Churn Status")
plt.xlabel("Credit Score Range")
plt.ylabel("Average Number of Products")

# Annotating
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom')

plt.legend(title="Churned (1 = Yes, 0 = No)")
plt.show()

print(df.columns)


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/53b659a7-02d9-4815-ab51-4c46ac7bc9dd" />

# Step 17: Satisfaction vs Points Earned by Churn

df['churn_status'] = df['exited'].map({1: 'Churned', 0: 'Retained'})

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='satisfaction_score',
    y='point_earned',
    hue='churn_status',

)
plt.title("Satisfaction vs Points Earned by Churn")
plt.xlabel("Satisfaction Score")
plt.ylabel("Points Earned")
plt.legend(title="Churn Status")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3a942eb2-e029-48e0-8157-08fa4a6bbf70" />

# Step 18 : Satisfaction Score Distribution by Churn
plt.figure(figsize=(8,6))
sns.histplot(
    data=df,
    x='satisfaction_score',
    hue='churn_status',
    bins=10,
    multiple='stack'
)
plt.title("Satisfaction Score Distribution by Churn")
plt.xlabel("Satisfaction Score")
plt.ylabel("Count")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/13638de7-6526-4245-8b78-b36174105a49" />

# Step 19: Churn Distribution Among Complainers
complain_df = df[df['complain'] == 1]
complain_churn = complain_df['churn_status'].value_counts(normalize=True)
plt.figure(figsize=(6,6))
plt.pie(
    complain_churn,
    labels=complain_churn.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Churn Distribution Among Complainers")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/13cc0a4b-fcb3-4d39-9066-2bb6933c9420" />

# Step 20: Salary vs Balance by Churn Status (sample 500 customers)
sample_df = df.sample(n=500, random_state=42)
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x='salary',
    y='balance',
    hue='churn_status',
    alpha=0.6
)
plt.title("Salary vs Balance by Churn Status")
plt.xlabel("Salary")
plt.ylabel("Balance")
plt.legend(title="Churn Status")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4e581c47-6146-4d0e-b511-f5bf94b76fc8" />

# Step 21: Average Balance by Number of Products and Churn Status
avg_balance = df.groupby(['numofproducts', 'churn_status'])['balance'].mean().reset_index()
plt.figure(figsize=(8,6))
sns.barplot(
    data=avg_balance,
    x='numofproducts',
    y='balance',
    hue='churn_status'
)
plt.title("Average Balance by Number of Products and Churn Status")
plt.xlabel("Number of Products")
plt.ylabel("Average Balance")
plt.legend(title="Churn Status")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/ee61b881-c828-4600-9f7b-f6e5d35ae049" />

# Step 22: Churn Distribution Among FD Holders
fd_df = df[df['hasfd'] == 1]
fd_churn = fd_df['churn_status'].value_counts(normalize=True).round(2)
plt.figure(figsize=(6,6))
plt.pie(
    fd_churn,
    labels=fd_churn.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Churn Distribution Among FD Holders")
plt.show()


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/d08608da-c5a5-42b4-872d-16f28d92f20c" />






















