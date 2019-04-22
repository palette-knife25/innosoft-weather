"""
Linear regression model
author: Alsu Vakhitova
"""
import datahelper
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

hour = timedelta(hours=1)

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

reg = LinearRegression().fit(x_train, y_train)


# R^2 is defined as (1 - u/v)
# where u is the residual sum of squares ((y_true - y_pred) ** 2).sum()
# and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
print("R^2 for all measurements: ", reg.score(x_test, y_test), '\n')

m = mean_absolute_error(y_test, reg.predict(x_test), multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(x.columns.values), m):
    print(col, ": ", err)
