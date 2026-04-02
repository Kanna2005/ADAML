# Load the necessary libraries
# R studio code
library(lubridate)
library(forecast)

electricity_data <- read.csv("C:/Users/Admin/Downloads/global_electricity_production_data.csv")

print(electricity_data)

electricity_data$Date <- as.Date(electricity_data$Date, format = "%m-%d-%Y") 

# Create a time series object
ts_data <- ts(electricity_data$Production, frequency = 365.25 / 7)


order_p <- 1  # Autoregressive order
order_d <- 1  # Differencing order
order_q <- 1  # Moving average order


# Fit the ARIMA model
arima_model <- arima(ts_data, order = c(order_p, order_d, order_q))


# Forecast future values
forecast_values <- forecast(arima_model, h = 20)  # You can adjust 'h' for the number of forecasted periods


# Print the forecasted values
print(forecast_values)


# Plot the forecasted values
plot(forecast_values, main = "Electricity Production Forecast",
     xlab = "Date", ylab = "Production")


#2] code: Python 

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
data = pd.read_csv("C:/Users/Admin/Downloads/global_electricity_production_data.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Convert Date column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format="%m-%d-%Y")

# Set Date as index
data.set_index('Date', inplace=True)

# Select Production column as time series
ts_data = data['Production']

# Define ARIMA model order (p, d, q)
order_p, order_d, order_q = 1, 1, 1

# Build and train ARIMA model
model = ARIMA(ts_data, order=(order_p, order_d, order_q))
arima_model = model.fit()

# Forecast next 20 steps (days)
forecast_steps = 20
forecast_values = arima_model.forecast(steps=forecast_steps)

# Print forecasted values
print("\nForecasted Values for Next", forecast_steps, "Days:")
print(forecast_values)

# Plot original data and forecast
plt.figure(figsize=(10, 5))
plt.plot(ts_data, label='Original Data')
plt.plot(forecast_values, label='Forecast (Next 20 Days)')

# Add labels and title
plt.title("Electricity Production Forecast")
plt.xlabel("Date")
plt.ylabel("Production")

# Show legend and grid
plt.legend()
plt.grid(True)

# Display plot
plt.show()


# Auto ML for Time Series
# Code:
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Load and preprocess data
file_path = "C:/Users/Admin/Desktop/Electric_Production.csv"
electricity_data = pd.read_csv(file_path)

# Convert DATE column to datetime
electricity_data['DATE'] = pd.to_datetime(
    electricity_data['DATE'],
    format="%m-%d-%Y"
)

# Set DATE as index
electricity_data.set_index('DATE', inplace=True)

# Select Value column
ts_data = electricity_data['Value']

# Step 2: Train ARIMA model (1,1,1)
model = ARIMA(ts_data, order=(1, 1, 1))
model_fit = model.fit()

# Step 3: Generate forecast
forecast_values = model_fit.forecast(steps=20)

# Step 4: Display forecast
print("----- Forecasted Values -----")
print(forecast_values)

# Step 5: Visualization
plt.figure(figsize=(10, 6))

# Plot original data
plt.plot(ts_data, label="Original Data")

# Plot forecast
plt.plot(forecast_values, label="Forecast", linestyle="--")

# Labels and title
plt.title("Electricity Production Forecast")
plt.xlabel("Date")
plt.ylabel("Production Value")

# Legend
plt.legend()

# Show plot
plt.show()


