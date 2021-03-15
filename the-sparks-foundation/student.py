#Importing libraries
import numpy as np
import pandas as pd


#importing dataset
url = "http://bit.ly/w-data"
dataset = pd.read_csv(url)


#differentiating the dataset
x = dataset.iloc[ : , :-1 ].values
y = dataset.iloc[ : ,1].values

#differentiating in training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x , y , test_size = 1/5)


#fitting the training set
from sklearn.linear_model import LinearRegression     
regresor=LinearRegression()
regresor.fit(x_train, y_train)


#predicting the test set
predict = regresor.predict(x_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': predict})  
print(df)

#converting a string into a numpy array for prediction
hours = np.array(9.25)

#predicting the result
eg_pred = regresor.predict(hours.reshape(1, -1))
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(eg_pred[0]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, predict)) 