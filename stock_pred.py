def which_stock():
    print("---------------------------------------------")
    print("---------------------------------------------")
    print("--Which Stock price do you want to predict?--")
    print("---------------------------------------------")
    print("----1. Amazon--------------------------------")
    print("----2. Google--------------------------------")
    print("----3. Apple---------------------------------")
    print("----4. Netflix-------------------------------")
    print("---------------------------------------------")
    print("---------------------------------------------")
    choice = int(input("Enter choice(1-4) : "))
    global stock
    if choice == 1:
        stock = "AMZN"
    elif choice == 2:
        stock = "GOOG"
    elif choice == 3:
        stock = "AAPL"
    elif choice == 4:
        stock = "NFLX"
    else:
        print("Please enter a valid input!")
        which_stock()

def what_time():
    print("---------------------------------------------")
    print("---------------------------------------------")
    print("----Predictions for what time period?--------")
    print("---------------------------------------------")
    print("----1) Tomorrow------------------------------")
    print("----2) 1 month-------------------------------")
    print("----3) 3 months------------------------------")
    print("----4) 6 months------------------------------")
    print("---------------------------------------------")
    choice = int(input("Enter choice(1-4) : "))
    
    global timeper
    if choice == 1:
        timeper = 0
    elif choice == 2:
        timeper = 1
    elif choice == 3:
        timeper = 3
    elif choice == 4:
        timeper = 6
    else:
        print("Please enter a valid input!")
        what_time()

which_stock()
what_time()

import math
from sklearn.metrics import mean_squared_error
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from datetime import date

today = date.today()

df = web.DataReader(stock , data_source='yahoo', start='2016-01-01', end=str(today))

data = df.filter(['Close'])
dataset = data.values
training_data_len = math.ceil( len(dataset) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1)) 
scaled_data = scaler.fit_transform(dataset)

#
train_data = scaled_data[0:training_data_len  , : ]
y_test =  dataset[training_data_len : , : ]

x_train=[]
y_train = []
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

result = model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - 60: , : ]
#Create the x_test and y_test data sets
x_test = []

if timeper == 0:
    new_df = data.filter(['Close'])
    #Get teh last 60 day closing price 
    last_60_days = new_df[-60:].values
    #Scale the data to be values between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    X_test = []
    #Append teh past 60 days
    X_test.append(last_60_days_scaled)
    #Convert the X_test data set to a numpy array
    X_test = np.array(x_test)
    #Reshape the data
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #Get the predicted scaled price
    pred_price = model.predict(X_test)
    #undo the scaling 
    pred_price = scaler.inverse_transform(pred_price)
    print("Predicted closing price for tomorrow : ",pred_price)
if timeper in [1,3,6]:
    #Test data set
    test_data = scaled_data[training_data_len - 60: , : ]
    #Create the x_test and y_test data sets
    x_test = []
    y_test =  dataset[training_data_len : , : ] #Get all of the rows from index 1603 to the rest and all of the columns (in this case it's only column 'Close'), so 2003 - 1603 = 400 rows of data
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
    from pandas.tseries.offsets import DateOffset
    future_dates=[data.index[-1]+ DateOffset(months=x)for x in range(0,timeper)]
    future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)
    #concatenating
    future_df=pd.concat([df,future_datest_df])
    future_df['forecast'] = model.predict(x_test)  
    future_df[['Sales', 'forecast']].plot(figsize=(12, 8))


x_test = np.array(X_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
predictions = model.predict(X_test) 
predictions = scaler.inverse_transform(predictions)
#testScore = math.sqrt(mean_squared_error(y_test, predictions))
#print('RMSE Score: %.2f RMSE' % (testScore))

print('MAPE score : ',np.mean(np.abs((y_test - predictions) / y_test)) * 100)
