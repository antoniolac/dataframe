import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math

"""
  PART 1 DELATE OUTLIERS & STANDARDIZATION  
"""
#01-dataframe
df = pd.read_csv("pearson_dataset.csv")

#02-describe
print(df.describe())

#03-outlier
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_df = df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]

#04-data train and test split
train, test = train_test_split(filtered_df, test_size=0.2)
X_train = train.drop(columns=["sheight"])
y_train = train["sheight"]
X_test = test.drop(columns=["sheight"])
y_test = test["sheight"]

#05-standardization
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_std = scaler_X.fit_transform(X_train)
X_test_std = scaler_X.transform(X_test)
y_train_std = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_std = scaler_y.transform(y_test.values.reshape(-1, 1))

#06-model
model = LinearRegression()
model.fit(X_train_std, y_train_std)
y_pred = model.predict(X_test_std)

#07-rmse
rmse1 = math.sqrt(mean_squared_error(y_test_std, y_pred))
print("\n----- PART 1 -----")
print("RMSE (no outlier, standardization):", rmse1)


"""
  PART 2 OUTLIER & NORMALIZATION  
"""
#01-dataset

#02-data train and test split
train2, test2 = train_test_split(df, test_size=0.2)
X_train2 = train2.drop(columns=["sheight"])
y_train2 = train2["sheight"]
X_test2 = test2.drop(columns=["sheight"])
y_test2 = test2["sheight"]

#03-normalization
minmax_X = MinMaxScaler()
minmax_y = MinMaxScaler()
X_train_norm = minmax_X.fit_transform(X_train2)
X_test_norm = minmax_X.transform(X_test2)
y_train_norm = minmax_y.fit_transform(y_train2.values.reshape(-1, 1))
y_test_norm = minmax_y.transform(y_test2.values.reshape(-1, 1))

#04-model
model2 = LinearRegression()
model2.fit(X_train_norm, y_train_norm)
y_pred2 = model2.predict(X_test_norm)

#05-rmse
rmse2 = math.sqrt(mean_squared_error(y_test_norm, y_pred2))
print("\n----- PART 2 -----")
print("RMSE (with outlier, normalization):", rmse2)

"""
  PART 3 SGDRegressor 
"""
#01-data part2

#02-SDGRegressor model
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train_norm, y_train_norm.ravel())
y_pred_sgd = sgd_model.predict(X_test_norm)

#03-rmse
rmse3 = math.sqrt(mean_squared_error(y_test_norm, y_pred_sgd))

#04-different parameters
sgd_model2 = SGDRegressor(max_iter=1500, tol=1e-4, eta0=0.01, learning_rate='constant')
sgd_model2.fit(X_train_norm, y_train_norm.ravel())
y_pred_sgd2 = sgd_model2.predict(X_test_norm)
rmse4 = math.sqrt(mean_squared_error(y_test_norm, y_pred_sgd2))

print("\n----- PART 3 -----")
print("RMSE SGDRegressor 1:", rmse3)
print("RMSE SGDRegressor 2:", rmse4)

"""
  PART 4 ScatterPlot
"""
plt.figure(figsize=(14, 6))

#ScatterPlot with outlier
plt.subplot(1, 2, 1)
plt.title("outliers & regression")
plt.scatter(X_test2, y_test2, color='red', label='Data with outlier')
plt.plot(X_test2, minmax_y.inverse_transform(y_pred2.reshape(-1, 1)), color='blue', label='LinearRegression')
plt.plot(X_test2, minmax_y.inverse_transform(y_pred_sgd.reshape(-1, 1)), color='green', linestyle='--', label='SGDRegressor')
plt.xlabel("pheight")
plt.ylabel("sheight")
plt.legend()

#ScatterPlot no outlier
plt.subplot(1, 2, 2)
plt.title("no outliers & regression")
plt.scatter(X_test, y_test, color='orange', label='Data no outlier')
plt.plot(X_test, scaler_y.inverse_transform(y_pred), color='blue', label='LinearRegression')
plt.xlabel("pheight")
plt.ylabel("sheight")
plt.legend()

plt.tight_layout()
plt.show()

"""
  FINAL COMPARISON
"""
print("\n----- FINAL COMPARISON -----")
print("RMSE Part 1 (no outlier, standardization):", rmse1)
print("RMSE Part 2 (with outlier, normalization):", rmse2)
print("RMSE Part 3 (SGDRegressor):", rmse3)
print("RMSE Part 3 (SGDRegressor 2):", rmse4)
