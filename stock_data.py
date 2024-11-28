import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Define the stock ticker symbol (e.g., Apple Inc.)
ticker = 'AAPL'

# Download historical data for the stock (last 10 years)
stock_data = yf.download(ticker, start="2014-01-01", end="2024-01-01")

# Display the first few rows of data
print(stock_data.head())

# Plot the stock's closing price
stock_data['Close'].plot(title=f'{ticker} Stock Price', figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.show()

# Use the 'Close' column for prediction
stock_data = stock_data[['Close']]

# Drop rows with missing values
stock_data.dropna(inplace=True)

# Shift the 'Close' column to create the target (y)
stock_data['Target'] = stock_data['Close'].shift(-1)

# Drop the last row (NaN in the target column)
stock_data.dropna(inplace=True)

print(stock_data.head())

# Feature (X) and Target (y)
X = stock_data[['Close']]  # Use 'Close' and '7_day_MA' as features
y = stock_data['Target']  # Next day's closing price as target

from sklearn.model_selection import train_test_split

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


print(f"Training data size: {len(X_train)}")
print(f"Test data size: {len(X_test)}")


from sklearn.ensemble import RandomForestRegressor

# Create the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("Predictions: ", predictions[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Calculate the Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")

# Plot the predictions vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue')
plt.plot(y_test.index, predictions, label='Predicted', color='red')
plt.title(f'{ticker} Stock Price Prediction (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Calculate R-squared score (coefficient of determination)
r_squared = model.score(X_test, y_test)

print(f"R-squared score: {r_squared}")
