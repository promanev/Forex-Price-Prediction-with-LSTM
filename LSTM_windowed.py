# LSTM for international airline passengers problem with window regression framing
# Original code taken from 
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# Courtesy of Jason Brownlee
# Modified by Roman Popov, 2019.
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set hyperparameters:
n_hidden = 200      # number of hidden units
look_back = 24     # Using 1 month worth of prices to predict next day's price
split_point = 0.67 # the proportion of the data to be used for training
epochs = 100       # how many time to repeat the training

# Convert an array of values into a dataset matrix:
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# Fix random seed for reproducibility:
numpy.random.seed(7)

# Load the dataset:
dataframe = read_csv('eurusd_1d_edit.csv', usecols=[2], engine='python', skipfooter=1)
dataset = dataframe.values
dataset = dataset.astype('float32')

# Normalize the dataset:
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split into train and test sets:
train_size = int(len(dataset) * split_point)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Reshape into X=t and Y=t+1:
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]:
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and fit the LSTM network
model = Sequential()

# Create a hidden layer with LSTM units:
model.add(LSTM(n_hidden, input_shape=(1, look_back)))
# Dropout setting aims to prevent overfitting by excluding a portion of inputs
# randomly. Fischer and Krauss (2017) reported optimal dropout of 0.1 when 
# using a LSTM network for stock price prediction:
model.add(Dropout(0.1))

# Can create additional layers:
# model.add(LSTM(n_hidden, input_shape=(1, look_back)))
# model.add(Dropout(0.1))

# Output layer with one neuron because predicting only tomorrow's price:
model.add(Dense(1))

# Compile the model:
model.compile(loss='mean_squared_error', optimizer='adam')

# Run training of the model:
model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=2)

# Make predictions:
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions so the error is sized as the original data:
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error:
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# Shift train predictions for plotting:
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting:
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# Plot baseline and predictions:
plt.figure(num=None, figsize=(16, 12), dpi=150, facecolor='w', edgecolor='k')
plt.plot(scaler.inverse_transform(dataset), label='Original data')
plt.plot(trainPredictPlot, label='Training data prediction')
plt.plot(testPredictPlot, label='Testing data prediction')
plt.legend()
plt.title('EUR/USD Forex Prices 2009-2018')
plt.show()
