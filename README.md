# prophet_prediction_model
Prophet Hyperparameter Optimization and Time Series Forecasting with Asset Price

Introduction
- This code performs time series forecasting using Facebook's Prophet library. It uses a grid search approach to optimize the hyperparameters of the model and then uses the best model to make predictions for the next 48 hours with a 1-minute frequency. The results are then plotted using matplotlib.

Prerequisites
- To run this code, the following dependencies must be installed:

- pandas
- matplotlib
- prophet
- sklearn
- Instructions

Install the required dependencies.
- Clone the repository and navigate to the directory containing the code.
- Run the code using a Python interpreter such as Jupyter Notebook or Python shell.

Code Explanation
- The code first imports the required libraries: pandas, matplotlib, Prophet from the Prophet library, and ParameterGrid from the sklearn library. It then defines the CSV file to read the data from and reads it into a pandas DataFrame.

- Next, the code preprocesses the data by selecting only the relevant columns, removing the timezone from the index, and splitting the data into train and test sets. The length of the training data and the forecasting horizon are defined, and the Prophet model is initialized.

- The hyperparameters to optimize are defined using a parameter grid. A grid search is then performed to find the best hyperparameters. The model is trained with the best hyperparameters, and predictions are made for the next 48 hours with a 1-minute frequency.

- Finally, the results are plotted using matplotlib. The plot is customized with a dark background, white tick labels and axis labels, and grey spines. The plot shows the actual data points and the predicted values for the next 48 hours.

Conclusion
- This code provides an example of how to perform time series forecasting using the Prophet library. The code can be easily adapted to work with most other datasets and can serve as a starting point for further analysis.

cbrwx
