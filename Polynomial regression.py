# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'manufacturing.csv'
manufacturing_df = pd.read_csv(file_path)

# ---------------------------
# Question 1: Data Loading and Initial Visualization
# ---------------------------
# (a) Load and display basic info
print(manufacturing_df.head())
print(manufacturing_df.info())

# (b) Scatter plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(manufacturing_df['Temperature (°C)'], manufacturing_df['Quality Rating'], color='blue')
plt.xlabel('Temperature (°C)')
plt.ylabel('Quality Rating')
plt.title('Temperature vs. Quality Rating')

# Temperature (°C) vs. Quality Rating
# Shape: The graph shows a very strong non-linear relationship.
#
# Trend:
#
# At lower temperatures (around 100°C to 270°C), the Quality Rating stays high (close to 100).
#
# Around 270–300°C, there’s a sudden and sharp drop in Quality Rating.
#
# Interpretation:
#
# The manufacturing process seems to perform well within a certain temperature range.
#
# After a critical temperature point (~275°C or higher), quality degrades rapidly — possibly due to overheating or material breakdown.
#
# Polynomial regression is more appropriate than linear regression for this data since the curve isn't a straight line.

plt.subplot(1, 2, 2)
plt.scatter(manufacturing_df['Pressure (kPa)'], manufacturing_df['Quality Rating'], color='green')
plt.xlabel('Pressure (kPa)')
plt.ylabel('Quality Rating')
plt.title('Pressure vs. Quality Rating')

plt.tight_layout()
plt.show()
#
# Pressure (kPa) vs. Quality Rating
# Shape: This plot looks very scattered and random.
#
# Trend:
#
# There’s no clear upward or downward pattern.
#
# Quality Rating values are widely spread for all Pressure values.
#
# Interpretation:
#
# Pressure (kPa) alone does not appear to have a strong or direct relationship with Quality Rating.
#
# The lack of pattern suggests that pressure might either:
#
# Have no effect, or
#
# Its effect might only be seen in combination with other variables (like temperature or material properties).
#
#  Polynomial regression may not help here since the data shows no visible pattern.


# (c) Analysis based on scatter plots: Add your own observations here

# ---------------------------
# Question 2: Linear vs. Quadratic Models for Temperature
# ---------------------------
X = manufacturing_df[['Temperature (°C)']]
y = manufacturing_df['Quality Rating']

# (a) Linear Regression (Degree 1)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

r2_lin = r2_score(y, y_pred_lin)
mse_lin = mean_squared_error(y, y_pred_lin)
print("Linear Model - R^2:", r2_lin)
print("Linear Model - MSE:", mse_lin)

# (b) Polynomial Regression (Degree 2)
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly2, y)
y_pred_quad = lin_reg2.predict(X_poly2)

r2_quad = r2_score(y, y_pred_quad)
mse_quad = mean_squared_error(y, y_pred_quad)
print("Quadratic Model - R^2:", r2_quad)
print("Quadratic Model - MSE:", mse_quad)

# (c) Plot actual and predictions
plt.scatter(X, y, color='gray', label='Actual')
plt.plot(X, y_pred_lin, color='blue', label='Linear Fit')
plt.plot(X, y_pred_quad, color='red', label='Quadratic Fit')
plt.xlabel('Temperature (°C)')
plt.ylabel('Quality Rating')
plt.title('Model Comparison: Linear vs. Quadratic')
plt.legend()
plt.show()

# Quadratic regression is the better choice for modeling the relationship between Temperature and Quality Rating.
#
# A higher-degree polynomial (e.g., cubic or 4th-degree) might even fit slightly better, but with the risk of overfitting.


# (d) Compare metrics and curve fit: Add your own interpretation here

# ---------------------------
# Question 3: Higher-Degree Polynomials & Overfitting
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
degrees = [1, 2, 3, 4, 5, 8]
r2_train_scores = []
r2_test_scores = []

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    r2_train = model.score(X_poly_train, y_train)
    r2_test = model.score(X_poly_test, y_test)
    r2_train_scores.append(r2_train)
    r2_test_scores.append(r2_test)

plt.plot(degrees, r2_train_scores, marker='o', label='Train R^2')
plt.plot(degrees, r2_test_scores, marker='x', label='Test R^2')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.title('Train vs Test R^2 for Varying Polynomial Degrees')
plt.legend()
plt.grid(True)
plt.show()

# (d) Analyze overfitting trends based on plot: Add your own interpretation here

# ---------------------------
# Question 4: Cross-Validation for Model Selection
# ---------------------------
avg_mse_scores = []
deg_range = range(1, 7)

for d in deg_range:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=d)),
        ('linreg', LinearRegression())
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_squared_error')
    avg_mse = -scores.mean()
    avg_mse_scores.append(avg_mse)

plt.plot(deg_range, avg_mse_scores, marker='s')
plt.xlabel('Polynomial Degree')
plt.ylabel('Avg Cross-Validated MSE')
plt.title('Cross-Validation MSE vs Polynomial Degree')
plt.grid(True)
plt.show()

# (c) Choosebest degree based on lowest MSE: Add your justification here

# ---------------------------
# Question 5: Final Model and Prediction
# ---------------------------
best_degree = avg_mse_scores.index(min(avg_mse_scores)) + 1
poly_final = PolynomialFeatures(degree=best_degree)
X_poly_final = poly_final.fit_transform(X)
model_final = LinearRegression()
model_final.fit(X_poly_final, y)

# (b) Predict quality for Temperature = 215°C
X_new = poly_final.transform([[215]])
predicted_quality = model_final.predict(X_new)
print(f"Predicted Quality Rating for 215°C: {predicted_quality[0]:.2f}")

# (c) Interpret model curve: Add your thoughts here

# (d) Limitations: Add discussion on limitations such as extrapolation and ignoring pressure
