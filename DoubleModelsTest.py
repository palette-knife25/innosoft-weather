from training import datahelper
import numpy as np
import pickle
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

columns = [y.columns.get_loc("snow_intensity"), y.columns.get_loc("rain_intensity")]

# Training snow prediction
huber1 = HuberRegressor().fit(x_train.values, y_train.values[:,y.columns.get_loc("snow_intensity")])
#filename = 'snow_huber.sav'
#pickle.dump(huber1, open(filename, 'wb'))

# Prediction
# huber1 = pickle.load(filename)
y_pred = np.zeros((y_test.shape[0], 1))
pred1 = np.array([huber1.predict(x_test.values)]).T
y_pred = np.hstack((y_pred, pred1))
y_pred = y_pred[:,1:]

# Training rain prediction
huber2 = HuberRegressor().fit(x_train.values, y_train.values[:,y.columns.get_loc("rain_intensity")])
#filename = 'rain_huber.sav'
#pickle.dump(huber1, open(filename, 'wb'))

# Prediction
# huber2 = pickle.load(filename)
pred2 = np.array([huber2.predict(x_test.values)]).T
y_pred = np.hstack((y_pred, pred2))


# Training temperature prediction
extra_trees = ExtraTreesRegressor(n_jobs=-1, n_estimators=100)
extra_trees.fit(x_train, y_train)
#filename = 'temp_et.sav'
#pickle.dump(extra_trees, open(filename, 'wb'))

# Predicition
# extra_trees = pickle.load(filename)
y_predicted = extra_trees.predict(x_test)

# Replacing columns "rain_intensity" and "snow_intensity" in ET presictions
# by Huber predictions
y_predicted[:, y.columns.get_loc("snow_intensity")] = y_pred[:, 0]
y_predicted[:, y.columns.get_loc("rain_intensity")] = y_pred[:, 1]


# Evaluation
print('R^2 = ' + str(r2_score(y_test, y_predicted)))
print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_predicted))))
print()

m = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(y.columns.values), m):
    print(col, ": ", err)