"""
SVM regression model
author: Alsu Vakhitova
"""
import datahelper
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np


x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)

scalerX = StandardScaler()
scalerY = StandardScaler()

x_tr = scalerX.fit_transform(x)
y_tr = scalerY.fit_transform(y)

m = list()
y_pred = np.zeros((y_tr.shape[0], 1))

# SVR predicts for one feature at a time => iterate through columns
for column in range(y_tr.shape[1]):
    svr = SVR(kernel='rbf', degree=3, gamma='auto_deprecated',
              coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
              cache_size=200, verbose=False, max_iter=-1).fit(x_tr,y_tr[:,column])
    pred = np.array([svr.predict(x_tr)]).T
    y_pred = np.hstack((y_pred, pred))

y_pred = scalerY.inverse_transform(y_pred[:,1:])  # un-scaling back to original

m = mean_absolute_error(y, y_pred, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(y.columns.values), m):
    print(col, ": ", err)