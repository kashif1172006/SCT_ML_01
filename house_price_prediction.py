import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv('train.csv')

# Extract exactly the features asked for, plus the target price
# GrLivArea = Square Footage | BedroomAbvGr = Bedrooms | FullBath = Bathrooms
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]

# Handle any missing values by filling them with the median
X = X.fillna(X.median())

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("--- Model Evaluation ---")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}\n")

# Test it with a custom house scenario
# Let's predict a house with 2,000 sqft, 3 bedrooms, and 2 bathrooms
custom_house = pd.DataFrame([[2000, 3, 2]], columns=features)
predicted_price = model.predict(custom_house)

print("--- Custom Prediction ---")
print(f"Predicted price for a 2,000 sqft, 3-bed, 2-bath house: ${predicted_price[0]:,.2f}") 