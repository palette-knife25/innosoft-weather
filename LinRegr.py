"""
Linear regression model
author: Alsu Vakhitova
"""
import datahelper
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import numpy as np

hour = timedelta(hours=1)

x, y = datahelper.get_xy('51527.json', num_hours=3, error_minutes=15)

reg = LinearRegression().fit(x, y)

# R^2 is defined as (1 - u/v)
# where u is the residual sum of squares ((y_true - y_pred) ** 2).sum()
# and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum()
print("R^2 for all measurements: ", reg.score(x, y), '\n')

m = mean_absolute_error(y, reg.predict(x), multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(x.columns.values), m):
    print(col, ": ", err)
