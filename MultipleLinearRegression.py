import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. Generate Synthetic Dataset
np.random.seed(42)
n_samples = 200
enginesize = np.random.normal(2.5, 0.8, n_samples) * 100
curbweight = np.random.normal(3000, 400, n_samples)
horsepower = np.random.normal(150, 30, n_samples)
citympg = np.random.normal(25, 5, n_samples)
highwaympg = citympg + np.random.normal(3, 1.5, n_samples)

price = (
    200 * enginesize +
    0.3 * curbweight +
    120 * horsepower -
    100 * citympg -
    80 * highwaympg +
    np.random.normal(0, 1000, n_samples)
)

df = pd.DataFrame({
    'enginesize': enginesize,
    'curbweight': curbweight,
    'horsepower': horsepower,
    'citympg': citympg,
    'highwaympg': highwaympg,
    'price': price
})

# 2. Split Dataset
X = df[['enginesize', 'curbweight', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 3. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("Hypothesis Function:")
print(f"Price = {model.intercept_:.2f} + " + " + ".join([f"{coef:.2f} * {col}" for coef, col in zip(model.coef_, X.columns)]))
print("\nTrain MSE:", mean_squared_error(y_train, y_train_pred))
print("Train MAE:", mean_absolute_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Test MAE:", mean_absolute_error(y_test, y_test_pred))
print("R² Score on Test Set:", r2_score(y_test, y_test_pred))

# 5. Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()





# Q2: Manual Gradient Descent (Without Vectorization)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate synthetic dataset (same as Q1)
np.random.seed(42)
n_samples = 200
enginesize = np.random.normal(2.5, 0.8, n_samples) * 100
curbweight = np.random.normal(3000, 400, n_samples)
horsepower = np.random.normal(150, 30, n_samples)
citympg = np.random.normal(25, 5, n_samples)
highwaympg = citympg + np.random.normal(3, 1.5, n_samples)

price = (
        200 * enginesize +
        0.3 * curbweight +
        120 * horsepower -
        100 * citympg -
        80 * highwaympg +
        np.random.normal(0, 1000, n_samples)
)

df = pd.DataFrame({
    'enginesize': enginesize,
    'curbweight': curbweight,
    'horsepower': horsepower,
    'citympg': citympg,
    'highwaympg': highwaympg,
    'price': price
})


# Normalize features for better gradient descent convergence
def normalize(X):
    return (X - X.mean()) / X.std()


features = ['enginesize', 'curbweight', 'horsepower', 'citympg', 'highwaympg']
X = normalize(df[features])
y = df['price'].values.reshape(-1, 1)

# Add bias column (x0 = 1)
X.insert(0, 'bias', 1)
X = X.values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Initialize weights
n_features = X_train.shape[1]
theta = np.zeros((n_features, 1))
alpha = 0.01
iterations = 1000
m = len(y_train)

# Gradient Descent (non-vectorized)
cost_history = []

for i in range(iterations):
    predictions = np.dot(X_train, theta)
    error = predictions - y_train

    # Manual loop for gradients (no vectorization)
    gradients = np.zeros_like(theta)
    for j in range(n_features):
        for k in range(m):
            gradients[j] += (1 / m) * error[k][0] * X_train[k][j]

    theta -= alpha * gradients

    # Cost (MSE)
    cost = np.sum((predictions - y_train) ** 2) / (2 * m)
    cost_history.append(cost)

# Final learned weights
weights = theta

# Plot cost function over iterations
plt.plot(range(iterations), cost_history, color='purple')
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function over Iterations (Non-Vectorized Gradient Descent)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict on test set
y_pred_test = np.dot(X_test, weights)
test_mse = mean_squared_error(y_test, y_pred_test)

# Results
print("Final Learned Weights:")
for i, w in enumerate(weights):
    print(f"Theta {i}: {w[0]:.4f}")
print("\nFinal Test MSE:", test_mse)


#
# Task:
# Use the same Car Price Dataset to predict both citympg and highwaympg using the features:
#
# enginesize, curbweight, horsepower, and price
#
# We'll implement a vectorized multivariate linear regression for this.
#  Multivariate Linear Regression (Predicting Two Targets)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Generate Synthetic Dataset
np.random.seed(42)
n_samples = 200
enginesize = np.random.normal(2.5, 0.8, n_samples) * 100
curbweight = np.random.normal(3000, 400, n_samples)
horsepower = np.random.normal(150, 30, n_samples)
citympg = np.random.normal(25, 5, n_samples)
highwaympg = citympg + np.random.normal(3, 1.5, n_samples)
price = (
    200 * enginesize +
    0.3 * curbweight +
    120 * horsepower -
    100 * citympg -
    80 * highwaympg +
    np.random.normal(0, 1000, n_samples)
)

df = pd.DataFrame({
    'enginesize': enginesize,
    'curbweight': curbweight,
    'horsepower': horsepower,
    'price': price,
    'citympg': citympg,
    'highwaympg': highwaympg
})

# 2. Feature Selection
X = df[['enginesize', 'curbweight', 'horsepower', 'price']]
Y = df[['citympg', 'highwaympg']]  # Multivariate target

# 3. Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

# 4. Model Training
model = LinearRegression()
model.fit(X_train, Y_train)

# 5. Predictions
Y_pred = model.predict(X_test)

# 6. Evaluation (MSE for both outputs)
mse_city = mean_squared_error(Y_test['citympg'], Y_pred[:, 0])
mse_highway = mean_squared_error(Y_test['highwaympg'], Y_pred[:, 1])

print(f"Mean Squared Error (City MPG): {mse_city:.4f}")
print(f"Mean Squared Error (Highway MPG): {mse_highway:.4f}")

# 7. Plot Actual vs Predicted
plt.figure(figsize=(12, 5))

# City MPG
plt.subplot(1, 2, 1)
plt.scatter(Y_test['citympg'], Y_pred[:, 0], color='blue', alpha=0.6)
plt.plot([Y_test['citympg'].min(), Y_test['citympg'].max()],
         [Y_test['citympg'].min(), Y_test['citympg'].max()], 'r--')
plt.xlabel("Actual City MPG")
plt.ylabel("Predicted City MPG")
plt.title("Actual vs Predicted: City MPG")

# Highway MPG
plt.subplot(1, 2, 2)
plt.scatter(Y_test['highwaympg'], Y_pred[:, 1], color='green', alpha=0.6)
plt.plot([Y_test['highwaympg'].min(), Y_test['highwaympg'].max()],
         [Y_test['highwaympg'].min(), Y_test['highwaympg'].max()], 'r--')
plt.xlabel("Actual Highway MPG")
plt.ylabel("Predicted Highway MPG")
plt.title("Actual vs Predicted: Highway MPG")

plt.tight_layout()
plt.show()
