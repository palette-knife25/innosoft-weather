import datahelper
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

x, y = datahelper.get_xy('52712.json', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

gradient_boost = GradientBoostingRegressor(learning_rate=0.1)
model = MultiOutputRegressor(estimator=gradient_boost, n_jobs=-1)

estimators = np.arange(10, 1000, 10)
scores = dict()
current_index = 0
for n in estimators:
    model.set_params(estimator=GradientBoostingRegressor(n_estimators=n, learning_rate=0.1))
    model.fit(x_train, y_train)
    scores[current_index] = model.score(x_test, y_test)
    current_index += 1

sorted_by_scores = [(k, scores[k]) for k in sorted(scores, key=scores.get, reverse=True)]

print('Results of 5 estimators giving best results:\n')

for i in range(0, 5):
    index, score = sorted_by_scores[i]
    print("Number of estimators = ", estimators[index])

    model.set_params(estimator=GradientBoostingRegressor(n_estimators=estimators[index], learning_rate=0.1))
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)

    print('R^2 = ' + str(r2_score(y_test, y_predicted)))
    print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_predicted))))
    print()

    m = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
    print('Average mean absolute error: ', np.average(m))
    print("Mean absolute error for measurements:")
    for col, err in zip(list(x_test.columns.values), m):
        print(col, ": ", err)
    print()


