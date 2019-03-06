"""
KNN regression model
author: Alsu Vakhitova
"""
import datahelper
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


x, y = datahelper.get_xy('51527.json', num_hours=3, error_minutes=15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for n in range(1,10):
    print("\nNumber of neighbors: ", n)
    neigh = KNeighborsRegressor(n_neighbors=n)
    neigh.fit(x_train, y_train)

    print("R^2 for all measurements: ", neigh.score(x_test, y_test), '\n')

    m = mean_absolute_error(y_test, neigh.predict(x_test), multioutput='raw_values')
    print('Average mean absolute error: ', np.average(m))
    print("Mean absolute error for measurements:")
    for col, err in zip(list(x.columns.values), m):
        print(col, ": ", err)