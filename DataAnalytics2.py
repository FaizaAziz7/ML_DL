import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris

# Set seaborn style
sns.set(style="whitegrid")

# -------------------------------
# Task 1: Customer Spending Distribution
# -------------------------------
# Simulated monthly spending data for 500 customers
np.random.seed(42)
spending = np.random.gamma(shape=2.0, scale=50.0, size=500)  # Skewed distribution

plt.figure(figsize=(10, 5))
sns.histplot(spending, kde=True, bins=30, color='skyblue')
plt.title('Customer Monthly Spending Distribution')
plt.xlabel('Monthly Spending ($)')
plt.ylabel('Number of Customers')
plt.axvline(np.mean(spending), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# Insights:
# - Histogram + KDE shows right-skewed distribution.
# - Most customers spend on the lower end; few high spenders are outliers.

# -------------------------------
# Task 2: Exam Scores Analysis
# -------------------------------
# Simulated exam scores of 100 students
scores = np.random.normal(loc=75, scale=10, size=100)
scores = np.clip(scores, 0, 100)  # Ensuring scores between 0-100

plt.figure(figsize=(10, 5))
sns.boxplot(x=scores, color='lightgreen')
plt.title('Exam Score Distribution')
plt.xlabel('Scores')
plt.show()

plt.figure(figsize=(10, 5))
sns.histplot(scores, kde=True, color='orange', bins=15)
plt.title('Histogram of Exam Scores')
plt.xlabel('Score')
plt.ylabel('Number of Students')
plt.axvline(np.mean(scores), color='red', linestyle='--', label='Mean')
plt.legend()
plt.show()

# Insights:
# - Histogram suggests approximate normal distribution.
# - Boxplot can highlight potential outliers or skewness (if present).

# -------------------------------
# Task 3: Product Review Ratings
# -------------------------------
# Simulated review ratings (1 to 5 stars)
ratings = np.random.choice([1, 2, 3, 4, 5], size=300, p=[0.05, 0.1, 0.2, 0.4, 0.25])

plt.figure(figsize=(8, 5))
sns.countplot(x=ratings, hue=ratings, palette='Set2', legend=False)
plt.title('Product Review Ratings')
plt.xlabel('Rating (Stars)')
plt.ylabel('Number of Reviews')
plt.show()

# Insights:
# - Bar plot shows most users gave 4 stars, indicating overall good satisfaction.
# - Few 1-star ratings suggest few dissatisfied customers.

# -------------------------------
# Task 4: Study Hours vs. Exam Scores
# -------------------------------
# Simulated data
np.random.seed(10)
study_hours = np.random.uniform(0, 10, 100)
scores = study_hours * 7 + np.random.normal(0, 5, 100)  # Positive correlation with noise

plt.figure(figsize=(8, 5))
sns.scatterplot(x=study_hours, y=scores, color='blue')
sns.regplot(x=study_hours, y=scores, scatter=False, color='red')  # Trend line
plt.title('Study Hours vs. Exam Scores')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.show()

# Insight:
# - Clear **positive correlation**: more study hours generally lead to higher scores.

# -------------------------------
# Task 5: Advertising Budget vs. Sales
# -------------------------------
# Simulated data for 3 months (30 days each)
days = np.arange(1, 31)
ad_budget = np.random.uniform(1000, 5000, 30)
sales = ad_budget * 0.4 + np.random.normal(0, 200, 30)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=ad_budget, y=sales, color='green')
sns.regplot(x=ad_budget, y=sales, scatter=False, color='darkgreen')
plt.title('Advertising Budget vs. Sales')
plt.xlabel('Advertising Budget ($)')
plt.ylabel('Sales ($)')
plt.show()

# Insight:
# - Positive trend: higher ad budget generally leads to better sales, though some variability exists.

# -------------------------------
# Task 6: Age vs. Blood Sugar Level
# -------------------------------
# Simulated data for 100 patients
age = np.random.randint(20, 80, 100)
blood_sugar = 70 + (age * 0.5) + np.random.normal(0, 10, 100)  # Slight upward trend

plt.figure(figsize=(8, 5))
sns.scatterplot(x=age, y=blood_sugar, color='purple')
sns.regplot(x=age, y=blood_sugar, scatter=False, color='red')
plt.title('Age vs. Blood Sugar Level')
plt.xlabel('Age (years)')
plt.ylabel('Blood Sugar Level (mg/dL)')
plt.show()

# Insight:
# - Weak **positive trend**: older patients may have slightly higher blood sugar.
# - Some **outliers** or anomalies might be present.

# -------------------------------
# Task 7: House Price Prediction
# -------------------------------
# Simulated house data
np.random.seed(1)
house_size = np.random.randint(1000, 3500, 100)
bedrooms = np.random.randint(1, 6, 100)
price = house_size * 150 + bedrooms * 10000 + np.random.normal(0, 20000, 100)

# 3D Scatter Plot
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(house_size, bedrooms, price, c=price, cmap='viridis')
ax.set_xlabel('House Size (sq ft)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price ($)')
plt.title('3D Scatter: House Size vs Bedrooms vs Price')
plt.show()

# Alternatively, a pairplot
df_house = pd.DataFrame({'Size': house_size, 'Bedrooms': bedrooms, 'Price': price})
sns.pairplot(df_house)
plt.suptitle("Pair Plot: House Features vs Price", y=1.02)
plt.show()

# -------------------------------
# Task 8: Car Attributes Analysis
# -------------------------------
# Simulated car data
car_df = pd.DataFrame({
    'Horsepower': np.random.randint(70, 250, 100),
    'Weight': np.random.randint(1500, 4000, 100),
    'Fuel_Efficiency': np.random.uniform(10, 35, 100),
    'Engine_Size': np.random.uniform(1.0, 5.0, 100)
})

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(car_df.corr(), annot=True, cmap='coolwarm')
plt.title('Car Attributes Correlation Heatmap')
plt.show()

# Optional pairplot
sns.pairplot(car_df)
plt.suptitle("Car Attributes Pair Plot", y=1.02)
plt.show()

# -------------------------------
# Task 9: Student Performance Analysis
# -------------------------------
# Simulated data
study_hours = np.random.uniform(0, 10, 100)
sleep_hours = np.random.uniform(4, 10, 100)
exam_scores = study_hours * 7 + sleep_hours * 3 + np.random.normal(0, 5, 100)

student_df = pd.DataFrame({
    'Study Hours': study_hours,
    'Sleep Hours': sleep_hours,
    'Exam Score': exam_scores
})

# Multivariate scatter with hue
plt.figure(figsize=(8, 6))
sns.scatterplot(data=student_df, x='Study Hours', y='Sleep Hours', hue='Exam Score', palette='coolwarm', size='Exam Score')
plt.title('Study vs Sleep vs Exam Score')
plt.show()

# -------------------------------
# Task 10: Iris Flower Dataset
# -------------------------------
# Load Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Pairplot
sns.pairplot(iris_df, hue="species", corner=True)
plt.suptitle("Iris Dataset Pair Plot", y=1.02)
plt.show()
