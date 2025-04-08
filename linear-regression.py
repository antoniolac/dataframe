import pandas as pd 

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("pearson_dataset.csv")

import matplotlib.pyplot as plt

# y = mx + q
"""x = df.drop(columns = ["sheight"]) #solo colanna padre
y = df["sheight"] #colonna target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("dimensione df di train", x_train.shape)
print("dimensione df di test", x_test.shape)

#standardScaler
scaler = StandardScaler()

df2 = pd.DataFrame()
scaler.fit(x_train)
scaler.fit(y_train)

x_train_std = scaler.transform(x_train)
y_train_std = scaler.transform(y_train)

x_test_std = scaler.transform(x_test)
y_test_std = scaler.transform(y_test)

model = LinearRegression()
model.fit(x_train_std, y_train_std)

print(model.intercept_)
print(model.coef_)

#fase di test
y_pred = model.predict(x_test_std)

rmse= root_mean_squared_error(y_test, y_pred)

print(rmse)"""

train, test = train_test_split(df, test_size=0.2)

train_df = pd.DataFrame(train, columns= df.columns)
test_df = pd.DataFrame(test, columns= df.columns)

train_df.info()

scaler = StandardScaler()
scaler.fit(train_df)

train_df_std = scaler.transform(train_df)
test_df_std = scaler.transform(test_df)

model = LinearRegression()
model.fit(test_df_std)

print(model.intercept_)
print(model.coef_)

#fase di test
y_pred = model.predict(train_df)

rmse= root_mean_squared_error(test_df, y_pred)

print(rmse)

plt.boxplot(test_df, orientation = "horizontal")
plt.show()

plt.boxplot(test_df_std, orientation = "horizontal")
plt.show()

def plot_scatter_with_line(df, x_col, y_col, model, x_label = None, y_label = None, title = None):
    plt.figure(figsize=(10,6))
    plt.scatter(df[x_col], df[y_col], label='Data', color='blue', alpha
                = 0.5)
    plt.plot(df[x_col], model.predict(df[[x_col]]), color='red', linewidth=
             3, label='Regression Line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    
plot_scatter_with_line(df, "fheight", "sheight", model)


