import datahelper
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

mlp = MLPRegressor(hidden_layer_sizes=(11, 11, 11), max_iter=10000)

mlp.fit(x_train, y_train)

predicted = mlp.predict(x_test)


print('R^2 = ' + str(r2_score(y_test, predicted)))
print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, predicted))))
print()

m = mean_absolute_error(y_test, predicted, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(x_test.columns.values), m):
    print(col, ": ", err)