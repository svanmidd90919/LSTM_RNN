'''
Created on 2019 M10 31

@author: Sheldon Van Middelkoop
'''
"""
-----------------------------------------------------------------------------------------------------------------------
Importing Libraries
-----------------------------------------------------------------------------------------------------------------------
"""
import numpy
from pandas import read_csv
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

"""
-----------------------------------------------------------------------------------------------------------------------
Read training set data and preprocess for Deep Neural Network
-----------------------------------------------------------------------------------------------------------------------
"""
series = read_csv('train.csv', delimiter=',')

series = series[(series['date'] >= '2013-01-01')]

# Select New England States
series = series[(series['store'] == 2) & (series['item'] == 3)]
print(series[['date', 'sales']])

series = series[['sales']]

series = series.values
series = series.astype('float32')


"""
-----------------------------------------------------------------------------------------------------------------------
Normalize the data
-----------------------------------------------------------------------------------------------------------------------
"""

scaler = MinMaxScaler(feature_range=(0,1))

series = scaler.fit_transform(series)

print(series)

"""
-----------------------------------------------------------------------------------------------------------------------
Split into train and testing sets
-----------------------------------------------------------------------------------------------------------------------
"""
train_size = int(len(series) * 0.8)
test_set = int(len(series)) - train_size
train, test = series[0:train_size,:], series[train_size:len(series),:]

print(len(train), len(test))

"""
-----------------------------------------------------------------------------------------------------------------------
Properly vectorize the data
-----------------------------------------------------------------------------------------------------------------------
"""

def Dataset(series, prev=1):
    X, Y = [],[]
    for i in range(len(series) - prev - 1):
        val = series[i : ( i + prev ), 0]
        X.append(val)
        Y.append(series[i + prev, 0])
    return numpy.array(X), numpy.array(Y)

prev = 1
X_train , Y_train = Dataset(series, prev)
X_test, Y_test = Dataset(series, prev)

"""
-----------------------------------------------------------------------------------------------------------------------
Reshape data
-----------------------------------------------------------------------------------------------------------------------
"""

X_train = numpy.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = numpy.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

"""
-----------------------------------------------------------------------------------------------------------------------
Create and fit LSTM
-----------------------------------------------------------------------------------------------------------------------
"""

model = Sequential()
model.add(LSTM(4, input_shape=(1, prev)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

"""
-----------------------------------------------------------------------------------------------------------------------
Make predictions
-----------------------------------------------------------------------------------------------------------------------
"""

predictTrain = model.predict(X_train)
predictTest = model.predict(X_test)

# Now we must reverse scaling
predictTrain = scaler.inverse_transform(predictTrain)
trainY = scaler.inverse_transform([Y_train])

predictTest = scaler.inverse_transform(predictTest)
testY = scaler.inverse_transform([Y_test])

"""
-----------------------------------------------------------------------------------------------------------------------
Evaluate the predictions
-----------------------------------------------------------------------------------------------------------------------
"""
# Calculate root mean squared
trainEval = math.sqrt(mean_squared_error(trainY[0], predictTrain[:,0]))
print("Train RMSE Score: %.2f" % (trainEval))
testEval = math.sqrt(mean_squared_error(testY[0], predictTest[:,0]))
print("Test RMSE Score: %.2f" % (testEval))


trainPredictPlot = numpy.empty_like(series)
trainPredictPlot[:,:] = numpy.nan
trainPredictPlot[prev:len(predictTrain) + prev, :] = predictTrain
# Do the same for test data
testPredictPlot = numpy.empty_like(series)


"""
-----------------------------------------------------------------------------------------------------------------------
Plot the data
-----------------------------------------------------------------------------------------------------------------------
"""
plt.plot(scaler.inverse_transform(series))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
#Lets plot the data
#First the real training data

plt.subplot(2,1,1)
plt.scatter(X_test, testY, color='red')
plt.title("Test Data")
plt.ylabel("Sales of item 2 in store 3 (Real)")

# Then the predicted training data
plt.subplot(2,1,2)
plt.scatter(X_test, predictTest, color='green')
plt.ylabel("Sales of item 1 in store 1 (Predicted)")
plt.xlabel("Date - Integer Number of Days Since Start Date")

# Then the predicted training data
plt.subplot(2,2,1)
plt.scatter(X_train, trainY, color='magenta')
plt.ylabel("Sales of item 1 in store 1 (Predicted)")
plt.xlabel("Date - Integer Number of Days Since Start Date")

# Then the predicted training data
plt.subplot(2,2,2)
plt.scatter(X_train, predictTrain, color='cyan')
plt.ylabel("Sales of item 1 in store 1 (Predicted)")
plt.xlabel("Date - Integer Number of Days Since Start Date")

plt.show()

"""
-----------------------------------------------------------------------------------------------------------------------
--- End ---
-----------------------------------------------------------------------------------------------------------------------
"""