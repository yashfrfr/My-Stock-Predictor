#!/usr/bin/env python
# coding: utf-8

# In[56]:


#Import the libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# In[57]:


stock = pd.read_csv("C:/Users/suresh/Desktop/PYTHON ML DATA SCIENCE/Stock Prediction/Stock Data/ADANIPORTS.csv")


# In[58]:


stock.head()


# In[59]:


stock.info()


# In[60]:


stock.describe()


# In[61]:


# Ensure we are working with a copy of the DataFrame to avoid the SettingWithCopyWarning
stock = stock.copy()


# In[62]:


# Calculate HL_Perc and CO_Perc
stock.loc[:, 'HL_Perc'] = (stock['High'] - stock['Low']) / stock['Low'] * 100
stock.loc[:, 'CO_Perc'] = (stock['Close'] - stock['Open']) / stock['Open'] * 100


# In[63]:


dates = np.array(stock["Date"])
dates_check = dates[-30:]
dates = dates[:-30]


# In[64]:


stock.columns


# In[65]:


stock = stock[["HL_Perc", "CO_Perc", "Close", "Volume"]]


# In[66]:


#Define the label column
stock["PriceNextMonth"] = stock["Close"].shift(-30)


# In[67]:


stock.tail()


# In[68]:


#Make fetaure and label arrays
X = np.array(stock.drop(["PriceNextMonth"], 1))
X = preprocessing.scale(X)
X_Check = X[-30:]
X = X[:-30]
stock.dropna(inplace = True)
y = np.array(stock["PriceNextMonth"])


# In[69]:


#Divide the data set into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)


# In[70]:


#Define the prediction model
model = RandomForestRegressor()


# In[71]:


#Fit the model using training data
model.fit(X_train, y_train)


# In[72]:


#Calculate the confidence value by applying the model to testing data
conf = model.score(X_test, y_test)
print(conf)


# In[73]:


#Fit the model again using the whole data set
model.fit(X,y)


# In[74]:


predictions = model.predict(X_Check)


# In[75]:


#Make the final DataFrame containing Dates, ClosePrices, and Forecast values
actual = pd.DataFrame(dates, columns = ["Date"])
actual["ClosePrice"] = stock["Close"]
actual["Forecast"] = np.nan
actual.set_index("Date", inplace = True)
forecast = pd.DataFrame(dates_check, columns=["Date"])
forecast["Forecast"] = predictions
forecast["ClosePrice"] = np.nan
forecast.set_index("Date", inplace = True)
var = [actual, forecast]
result = pd.concat(var)  #This is the final DataFrame


# In[76]:


#Plot the final results
result.plot(figsize=(20,10), linewidth=1.5)
plt.legend(loc=2, prop={'size':20})
plt.xlabel('Date')
plt.ylabel('Price')


# In[ ]:




