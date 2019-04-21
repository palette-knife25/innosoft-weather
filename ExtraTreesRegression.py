import datahelper
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

extra_trees = ExtraTreesRegressor(n_jobs=-1)  # n_jobs=-1 is used for parallel calculations without limit

estimators = np.arange(10, 1000, 10)
scores = dict()
current_index = 0
for n in estimators:
    extra_trees.set_params(n_estimators=n)
    extra_trees.fit(x_train, y_train)
    scores[current_index] = extra_trees.score(x_test, y_test)
    current_index += 1

sorted_by_scores = [(k, scores[k]) for k in sorted(scores, key=scores.get, reverse=True)]

index, score = sorted_by_scores[0]
extra_trees.set_params(n_estimators=estimators[index])
extra_trees.fit(x_train, y_train)

filename = 'models/extra_trees_model.sav'
print('Saving model with ', estimators[index], ' estimators to file ', filename)
pickle.dump(extra_trees, open(filename, 'wb'))


# for i in range(0, 5):
#     index, score = sorted_by_scores[i]
#     print("Number of estimators = ", estimators[index])
#
#     extra_trees.set_params(n_estimators=estimators[index])
#     extra_trees.fit(x_train, y_train)
#     y_predicted = extra_trees.predict(x_test)
#
#     print('R^2 = ' + str(r2_score(y_test, y_predicted)))
#     print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_predicted))))
#     print()
#
#     m = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
#     print('Average mean absolute error: ', np.average(m))
#     print("Mean absolute error for measurements:")
#     for col, err in zip(list(x_test.columns.values), m):
#         print(col, ": ", err)
#     print()


