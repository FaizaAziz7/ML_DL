# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Combine students and grades DataFrames to include only students with grades
# Answer: Use inner join
students = pd.DataFrame({'student_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
grades = pd.DataFrame({'student_id': [1, 3], 'score': [85, 90]})
result_1 = pd.merge(students, grades, on='student_id', how='inner')
print(result_1)
# 2. Merge employee and department DataFrames ensuring all departments are shown
# Answer: Use right join
employees = pd.DataFrame({'emp_id': [1, 2], 'dept_id': [101, 102]})
departments = pd.DataFrame({'dept_id': [101, 102, 103], 'dept_name': ['HR', 'IT', 'Finance']})
result_2 = pd.merge(employees, departments, on='dept_id', how='right')
print(result_2)
# 3. Combine Spring, Summer, and Fall admissions vertically
# Answer: Use pd.concat
spring = pd.DataFrame({'name': ['Alice'], 'semester': ['Spring']})
summer = pd.DataFrame({'name': ['Bob'], 'semester': ['Summer']})
fall = pd.DataFrame({'name': ['Charlie'], 'semester': ['Fall']})
result_3 = pd.concat([spring, summer, fall], ignore_index=True)
print(result_3)
# 4. Merging two DataFrames but losing rows from first one
# Answer: Using inner join can cause this
left = pd.DataFrame({'id': [1, 2, 3]})
right = pd.DataFrame({'id': [1, 3]})
result_4 = pd.merge(left, right, on='id', how='inner')
print(result_4)
# 5. Total sales per product category
# Answer: Use groupby with sum
sales = pd.DataFrame({'category': ['A', 'B', 'A'], 'amount': [100, 200, 150]})
result_5 = sales.groupby('category')['amount'].sum()
print(result_5)
# 6. Average rating by each customer
# Answer: groupby + mean
reviews = pd.DataFrame({'customer': ['A', 'B', 'A'], 'rating': [5, 4, 3]})
result_6 = reviews.groupby('customer')['rating'].mean()
print(result_6)
# 7. Count how many students fall into grade brackets
# Answer: pd.cut + value_counts
scores = pd.DataFrame({'student': ['A', 'B', 'C', 'D'], 'score': [92, 75, 60, 88]})
scores['grade'] = pd.cut(scores['score'], bins=[0, 69, 79, 100], labels=['C', 'B', 'A'])
result_7 = scores['grade'].value_counts()
print(result_7)
# 8. Group ages into brackets
# Answer: pd.cut
ages = pd.DataFrame({'age': [15, 25, 40, 60]})
ages['age_group'] = pd.cut(ages['age'], bins=[0, 18, 35, 100], labels=['0-18', '19-35', '36+'])
print(ages['age_group'])
# 9. Categorize Celsius temps as 'Cold', 'Warm', 'Hot'
# Answer: Use pd.cut or custom function
temps = pd.DataFrame({'temp_c': [5, 20, 35]})
temps['label'] = pd.cut(temps['temp_c'], bins=[-float('inf'), 10, 25, float('inf')], labels=['Cold', 'Warm', 'Hot'])
print(temps['label'] )
# 10. Apply 10% discount
# Answer: Multiply by 0.9
products = pd.DataFrame({'Price': [100, 200, 300]})
products['Discounted'] = products['Price'] * 0.9
print(products['Discounted'])
# 11. Rename all columns to lowercase
# Answer: Use str.lower()
df = pd.DataFrame({'Name': ['A'], 'Age': [20]})
df.columns = df.columns.str.lower()
print(df.columns)
# 12. Save to CSV without index
# Answer: to_csv with index=False
print(df.to_csv('output.csv', index=False))

# 13. Visualize distribution of 'Income'
# Answer: Use plot.hist()
data = pd.DataFrame({'Income': [5000, 6000, 7000, 8000, 9000]})
data['Income'].plot.hist()
plt.title('Income Distribution')
plt.show()

# 14. Compare monthly sales of two products
# Answer: Use line plot
monthly_sales = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar'],
    'Product_A': [200, 220, 250],
    'Product_B': [180, 210, 260]
})
monthly_sales.set_index('Month').plot()
plt.title('Monthly Sales Comparison')
plt.show()

# 15. Relationship between math and science scores
# Answer: Use plot.scatter
performance = pd.DataFrame({'Math': [80, 90, 70], 'Science': [85, 95, 75]})
performance.plot.scatter(x='Math', y='Science')
plt.title('Math vs Science Scores')
plt.show()

# 16. Histogram + KDE for ages
# Answer: sns.histplot with kde=True
sns.histplot(data=ages, x='age', kde=True)
plt.title('Age Distribution with KDE')
plt.show()

# 17. Weight vs Height distribution
# Answer: sns.jointplot
weight_height = pd.DataFrame({'weight': [60, 70, 80], 'height': [150, 160, 170]})
sns.jointplot(data=weight_height, x='weight', y='height', kind='scatter')
plt.show()

# 18. Compare median income across job categories
# Answer: sns.boxplot
income_data = pd.DataFrame({
    'job': ['Engineer', 'Doctor', 'Artist', 'Doctor', 'Engineer'],
    'income': [70000, 90000, 50000, 95000, 72000]
})
sns.boxplot(data=income_data, x='job', y='income')
plt.title('Income Distribution by Job')
plt.show()

# 19. Response counts by gender
# Answer: sns.countplot
responses = pd.DataFrame({
    'gender': ['M', 'F', 'F', 'M', 'F'],
    'product': ['A', 'B', 'A', 'B', 'C']
})
sns.countplot(data=responses, x='product', hue='gender')
plt.title('Product Preferences by Gender')
plt.show()

# 20. Correlation matrix visualization
# Answer: sns.heatmap
df_corr = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})
sns.heatmap(df_corr.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
