"""
Bagging regression model
author: Alsu Vakhitova
"""
import datahelper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
import operator
import copy

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scores = {}
models = []

for n in range(2, 20):
    estimator = BaggingRegressor(max_samples=0.5, n_estimators=n)
    estimator.fit(x_train, y_train)
    scores[n] = estimator.score(x_test,y_test)
    models.append(copy.copy(estimator))

sorted_by_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
print('Results of 5 best # of estimators:\n')

for i in range(0, 5):
    n, score = sorted_by_scores[i]
    print("№ estimators = ", n)

    y_predicted = models[n-2].predict(x_test)

    print('R^2 = ' + str(r2_score(y_test, y_predicted)))
    print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_predicted))))
    print()

    m = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
    print('Average mean absolute error: ', np.average(m))
    print("Mean absolute error for measurements:")
    for col, err in zip(list(x_test.columns.values), m):
        print(col, ": ", err)
    print()