#Practical 2

# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Step 2: Load the dataset
df = pd.read_csv('sales_dataset.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)

# Step 3: Plot the time series data
plt.figure(figsize=(8,4))
plt.plot(df['Sales'], marker='o', label='Actual Sales')
plt.title('Monthly Sales Data (2020-2021)')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 4: Prepare data
df['Time'] = np.arange(len(df))
train = df.iloc[:-3]
test = df.iloc[-3:]

# Step 5: Linear Regression Model
lr = LinearRegression()
lr.fit(train[['Time']], train['Sales'])
pred_lr = lr.predict(test[['Time']])

# Step 6: Machine Learning Model (Random Forest)
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(train[['Time']], train['Sales'])
pred_rf = rf.predict(test[['Time']])

# Step 7: ARIMA Model
arima_model = ARIMA(train['Sales'], order=(1, 1, 1))
arima_fit = arima_model.fit()
pred_arima = arima_fit.forecast(steps=3)

# Step 8: Combine predictions
results = test.copy()
results['LR_Pred'] = pred_lr
results['RF_Pred'] = pred_rf
results['ARIMA_Pred'] = pred_arima.values

# Step 9: Evaluate models
def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

mae_lr, rmse_lr = evaluate(test['Sales'], results['LR_Pred'])
mae_rf, rmse_rf = evaluate(test['Sales'], results['RF_Pred'])
mae_arima, rmse_arima = evaluate(test['Sales'], results['ARIMA_Pred'])

# Step 10: Display comparison table
comparison = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'ARIMA'],
    'MAE': [mae_lr, mae_rf, mae_arima],
    'RMSE': [rmse_lr, rmse_rf, rmse_arima]
})

print("Model Performance Comparison:\n")
print(comparison)

# Step 11: Visual comparison
plt.figure(figsize=(8,4))
plt.plot(df.index, df['Sales'], label='Actual', marker='o')
plt.plot(results.index, results['LR_Pred'], label='Linear Regression', marker='x')
plt.plot(results.index, results['RF_Pred'], label='Random Forest', marker='^')
plt.plot(results.index, results['ARIMA_Pred'], label='ARIMA', marker='s')
plt.title('Model Comparison')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Step 12: Print results for observation
print("\nDetailed Predictions:\n")
print(results)


#conclusions:
'''
Conclusion: 
Based on the visual and numerical comparison, ARIMA gives the best
performance for time series forecasting. 
It accurately captures both the trend and time-based pattern of sales. Linear
Regression performs moderately well by fitting a trend line, while Random
Forest performs the weakest as it doesn’t consider time dependency. 

Hence, ARIMA is the most suitable model for predicting future sales in this
dataset.
'''

