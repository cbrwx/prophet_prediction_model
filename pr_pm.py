import pandas as pd
import matplotlib.pyplot as plt
try:
    import io, os, sys, setuptools, tokenize
    from prophet import Prophet
except ImportError:
    !pip install prophet
    import io, os, sys, setuptools, tokenize
    import prophet
    from prophet import Prophet
from sklearn.model_selection import ParameterGrid

# Define the CSV file and read it into a DataFrame
csv_file = 'your_downloaded_csv_file_from_yahoo_finance_with_historical_data.csv'
df = pd.read_csv(csv_file, parse_dates=['Datetime'], index_col='Datetime')

# Remove irrelevant columns from the DataFrame
df = df[['Close', 'Volume']]

# Remove the timezone from the index
df.index = df.index.tz_localize(None)

# Set the length of the training data and the length of the forecasting horizon
train_length = 168*60
forecast_length = 48*60

# Split the data into train and test sets
train_data = df.iloc[-train_length:]
test_data = df.iloc[-train_length-1:-train_length+forecast_length]

# Define the Prophet model
model = Prophet()

# Define the hyperparameters to optimize
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1],
    'seasonality_prior_scale': [0.01, 0.1, 1.0],
    'seasonality_mode': ['additive', 'multiplicative']
}

# Perform grid search optimization to find the best hyperparameters
grid = ParameterGrid(param_grid)
best_mape = None
for params in grid:
    model = Prophet(**params)
    model.fit(train_data.reset_index().rename(columns={'Datetime':'ds', 'Close':'y'}))
    forecast = model.predict(test_data.reset_index().rename(columns={'Datetime':'ds'}))
    mape = (abs(forecast['yhat'] - test_data['Close']) / test_data['Close']).mean()
    if best_mape is None or mape < best_mape:
        best_mape = mape
        best_params = params

# Train the model with the best hyperparameters
model = Prophet(**best_params)
model.fit(df.reset_index().rename(columns={'Datetime':'ds', 'Close':'y'}))

# Make predictions for the next 48 hours with 1-minute frequency
future = model.make_future_dataframe(periods=forecast_length, freq='1min')
forecast = model.predict(future)

# Plot graph
fig, ax = plt.subplots(figsize=(20, 8))  # set figure size
plt.style.use('dark_background')  # set the background color to black
colors = ['chartreuse', 'crimson', 'gold']  # define the colors for the plots

# Set style for the plot
ax.set_facecolor('grey')
ax.tick_params(colors='w', which='both', labelsize=16) # Modify tick label size for x and y axis
ax.yaxis.label.set_color('w')
ax.xaxis.label.set_color('w')
ax.spines['bottom'].set_color('w')
ax.spines['top'].set_color('w')
ax.spines['right'].set_color('w')
ax.spines['left'].set_color('w')
ax.grid(color='w', linestyle='-', linewidth=0.2)
# Set font size for x and y axis labels
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
model.plot(forecast, ax=ax)
ax.lines[1].set_color(colors[2])



