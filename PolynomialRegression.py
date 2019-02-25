import datahelper
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split


x, y = datahelper.get_xy('52712.json', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

polynomial_features = PolynomialFeatures(degree=3)
x_poly_train = polynomial_features.fit_transform(x_train)
x_poly_test = polynomial_features.fit_transform(x_test)

model = LinearRegression()
model.fit(x_poly_train, y_train)
y_poly_pred = model.predict(x_poly_test)


print('R^2 = ' + str(r2_score(y_test, y_poly_pred)))
print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_poly_pred))))
print()


m = mean_absolute_error(y_test, y_poly_pred, multioutput='raw_values')
print('Average mean absolute error: ', np.average(m))
print("Mean absolute error for measurements:")
for col, err in zip(list(x_test.columns.values), m):
    print(col, ": ", err)