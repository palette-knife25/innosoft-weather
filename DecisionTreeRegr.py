import datahelper
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import operator
import matplotlib.pyplot as plt

x, y = datahelper.get_xy('51527.json', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

scores = {}

for d in range(2,101):  # depths
    regr = DecisionTreeRegressor(max_depth=d)
    regr.fit(x_train, y_train)
    scores[d] = regr.score(x_test, y_test)

sorted_by_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

print('Results of 5 best depths:\n')

for i in range(0, 5):
    depth, score = sorted_by_scores[i]
    print("Depth = ", depth)

    regr = DecisionTreeRegressor(max_depth=depth)
    regr.fit(x_train, y_train)
    y_predicted = regr.predict(x_test)

    print('R^2 = ' + str(r2_score(y_test, y_predicted)))
    print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_predicted))))
    print()

    m = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
    print('Average mean absolute error: ', np.average(m))
    print("Mean absolute error for measurements:")
    for col, err in zip(list(x_test.columns.values), m):
        print(col, ": ", err)
    print()