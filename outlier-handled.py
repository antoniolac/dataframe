import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#01-dataset
df = pd.read_csv("pearson_dataset.csv")

#02-describe
print(df.describe())

#03-outlier

Q1 = np.array([df["fheight"].quantile(0.25), df["sheight"].quantile(0.25)])
Q3 = np.array([df["fheight"].quantile(0.75), df["sheight"].quantile(0.75)])
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

filtered_df = df[
    (df["fheight"] > lower_bound[0]) & (df["fheight"] < upper_bound[0]) &
    (df["sheight"] > lower_bound[1]) & (df["sheight"] < upper_bound[1])
]

#04-data train and test split
train, test = train_test_split(filtered_df, test_size=0.2, random_state=42)
# train e test sono array di  numpy

train = pd.DataFrame(train, columns=filtered_df.columns.values)
# train è un dataframe
test = pd.DataFrame(test, columns=filtered_df.columns.values)

X_train = train.drop(columns=["sheight"])
y_train = train['sheight']
X_test = test.drop(columns=["sheight"])
y_test = test['sheight']

#05-standardization
scaler = StandardScaler()
scaler.fit(train)
train_std = scaler.transform(train)

X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)
Y_train_std = scaler.transform(y_train)
Y_test_std = scaler.transform(y_test)

#06-model
model = LinearRegression()
model.fit(X_train_std, Y_train_std)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)

#07-rmse
y_pred = model.predict(X_test_std)
rmse = np.sqrt(mean_squared_error(Y_test_std, y_pred))
print("RMSE:", rmse)


