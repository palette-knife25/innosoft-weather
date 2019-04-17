"""
Huber regression model
author: Alsu Vakhitova
"""
import datahelper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

y_pred = np.zeros((y_test.shape[0], 1))
# Huber predicts for one feature at a time => iterate through columns
for column in range(y_test.shape[1]):
    svr = HuberRegressor().fit(x_train.values,y_train.values[:,column])
    pred = np.array([svr.predict(x_test.values)]).T
    y_pred = np.hstack((y_pred, pred))

y_pred = y_pred[:,1:]

print('R^2 = ' + str(r2_score(y_test, y_pred)))
print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print()

m = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(y.columns.values), m):
    print(col, ": ", err)