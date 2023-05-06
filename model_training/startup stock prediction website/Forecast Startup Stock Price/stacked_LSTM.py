import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from numpy import array
import matplotlib.pyplot as plt
import streamlit as st

st.title('Forecast Startup Stock')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = pdr.get_data_tiingo(user_input, api_key = 'cdaf52cefc8a7b480fd07bf9c5ebaddff7330127')

df.to_csv('AAPL.csv')

df = pd.read_csv('AAPL.csv')

df1 = df.reset_index()['close']

st.subheader('Data from 2018-2023')
st.write(df.describe())

scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data, test_data = df1[0:training_size,:],df1[training_size:len(df1),:1]

def create_dataset(dataset, time_step=1):
    dataX, dataY = [],[]
    for i in range (len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = load_model('stackedLSTM_model.h5')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# shift train predictions for plotting
look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back: len(train_predict) + look_back, :] = train_predict

# shift test predictions for plotting
st.subheader('Baseline and Predictions')
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back*2)+1:len(df1)-1, :] = test_predict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot, 'b')
plt.plot(testPredictPlot, 'g')
plt.show()

x_input = test_data[341:].reshape(1, -1)
x_input.shape

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

# demonstrate prediction for next 10 days

lst_output=[]
n_steps = 100
i = 0
while(i < 30):
    if(len(temp_input) > 100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose = 0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i+1

day_new = np.arange(1, 101)
day_pred = np.arange(101, 131)

df3 = df1.tolist()
df3.extend(lst_output)

st.subheader('Stock Price vs Time Chart with 30 days Predictions')
plt.plot(day_new, scaler.inverse_transform(df1[1159:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))

st.subheader('Stock Price vs Time Chart Combined with 30 days Predictions')
df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1000:])