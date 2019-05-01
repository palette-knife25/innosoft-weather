from training import datahelper
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split

x, y = datahelper.get_xy('data/', num_hours=3, error_minutes=15)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim = x_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(512, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(9, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()

NN_model.fit(x_train, y_train, epochs=500, batch_size=32, validation_split = 0.2, validation_data=(x_test, y_test))

filename = '../models/seq_nn_model.sav'
print('Saving model to file ', filename)
with open(filename, 'wb') as h:
    pickle.dump(NN_model, h)

# y_pred = NN_model.predict(x_test)
#
#
# print('R^2 = ' + str(r2_score(y_test, y_pred)))
# print('MSE = ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
# print()
#
# m = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
# print('Average mean absolute error: ', np.average(m))
# print("Mean absolute error for measurements:")
# for col, err in zip(list(y.columns.values), m):
#     print(col, ": ", err)