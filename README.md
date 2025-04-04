# PM Accelerator Data Science Assessment

Weather Trend Forecasting with Machine Learning

Overview

This project aims to forecast weather trends, specifically focusing on temperature, using machine learning and time series models. The dataset used is the Global Weather Repository, which contains daily weather information from various cities around the world. The project involves data cleaning, exploratory data analysis (EDA), model building, and evaluation, ultimately providing insights into how weather can be predicted using data science techniques.

Objective:
The objective of this project is to use the Global Weather Repository dataset to predict future temperature trends using machine learning and time series forecasting methods. The models used in this project include Linear Regression for a basic machine learning approach and Holt-Winters Exponential Smoothing for time series forecasting.

Dataset:
The dataset used in this project is the Global Weather Repository dataset, which provides daily weather information for cities worldwide. It includes various weather features such as:

Temperature (Celsius and Fahrenheit)

Precipitation

Wind Speed

Humidity

Atmospheric Pressure

Visibility

You can download the dataset from Kaggle:

Global Weather Repository - Kaggle

https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code

To run this project, you will need Python along with the following libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

statsmodels

scipy

Data Preprocessing:
Handling Missing Values

The dataset was checked for missing values. Any rows containing missing values were removed to ensure a clean dataset for analysis.

missing_values = df.isnull().sum()

df.dropna(inplace=True)

Normalization

To ensure the model performs efficiently, latitude and longitude were normalized using the MinMaxScaler to bring all numerical features onto a similar scale.

scaler = MinMaxScaler()

df[['latitude', 'longitude']] = scaler.fit_transform(df[['latitude', 'longitude']])

Outlier Detection:
Outliers in the temperature data were detected using Z-scores. Any data points with a Z-score greater than 2 were removed to ensure that extreme values don't skew the analysis.

df['temperature_zscore'] = zscore(df['temperature_celsius'])

df = df[df['temperature_zscore'].abs() <= 2]

Feature Engineering:
Time-based features were extracted from the last_updated column to use as inputs for the machine learning model:

Year

Month

Day

Hour

df['year'] = df.index.year

df['month'] = df.index.month

df['day'] = df.index.day

df['hour'] = df.index.hour

Exploratory Data Analysis (EDA):
EDA was performed to understand the data distribution and relationships between various weather features.

Temperature Distribution:
The distribution of temperatures was visualized using a histogram, allowing us to understand the overall spread of the temperature data.

sns.histplot(df['temperature_celsius'], bins=50, kde=True)

plt.title("Temperature Distribution")

plt.show()

Precipitation Distribution:
A similar histogram was plotted for precipitation to explore how much rainfall occurs in the dataset.

sns.histplot(df['precip_mm'], bins=50, kde=True)

plt.title("Precipitation Distribution")

plt.show()

Correlation Matrix:
A heatmap was created to visualize the correlation between different numerical weather features.

correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Correlation Matrix of Weather Features")

plt.show()

Modeling

Linear Regression:
The Linear Regression model was used to predict the temperature (temperature_celsius) based on time-based features and location-based features like latitude and longitude.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LinearRegression()

model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

Time Series Forecasting (Holt-Winters):
The Holt-Winters Exponential Smoothing method was used for time series forecasting. This method captures seasonality and trends in the data, which is important for predicting weather patterns over time.

holt_winters_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)

holt_winters_fit = holt_winters_model.fit()

predictions = holt_winters_fit.forecast(len(test))

Model Evaluation

Linear Regression Evaluation:
The Linear Regression model was evaluated using the following metrics:

Mean Squared Error (MSE): 63.86

Mean Absolute Error (MAE): 6.31

R²: 0.27

Holt-Winters Model Evaluation:
The Holt-Winters model was evaluated using the following metrics:

Mean Absolute Error (MAE): 82.77

Mean Squared Error (MSE): 9149.77

print("MAE:", mae_ts)

print("MSE:", mse_ts)

Visualizations:
Time Series Forecasting Visualization
A plot was created to compare the training data, actual data, and forecasted data from the Holt-Winters model.

plt.figure(figsize=(14, 7))

plt.plot(train.index, train, label='Training Data')

plt.plot(test.index, test, label='Actual Data')

plt.plot(test.index, predictions, label='Forecasted Data', color='red')

plt.title("Time Series Forecasting (Holt-Winters)")

plt.legend()

plt.show()

Insights & Conclusion
Key Insights:

The Linear Regression model has a low R² score (0.27), indicating that it does not capture the weather patterns effectively.

The Holt-Winters Exponential Smoothing model provided better performance, though there is still room for improvement, especially in terms of the MAE and MSE.

Conclusion:
This project demonstrates the application of machine learning and time series forecasting techniques to predict temperature trends. The Linear Regression model and the Holt-Winters method both have their strengths and weaknesses, and further refinement is needed for better performance.

Future Work:
Explore more advanced time series forecasting models such as ARIMA or Facebook Prophet.

Integrate additional weather features (e.g., humidity, wind speed) for more accurate predictions.

Perform hyperparameter tuning to improve model performance.
