#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# In[63]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.metrics import mean_squared_error


# In[3]:


car_data=pd.read_csv('car data.csv')
car_data.shape


# In[17]:


car_data.head()


# In[12]:


car_data.info()


# In[8]:


car_data.isnull().sum()


# In[22]:


car_data.dtypes


# In[10]:


car_data.describe().T


# #  categorical data analysis

# In[25]:


car_data['Fuel_Type'].value_counts()


# In[26]:


car_data.Seller_Type.value_counts()


# In[28]:


car_data.Transmission.value_counts()


# # Lebel Encoding

# In[36]:


# encoding "Fuel_Type" Column
car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[37]:


car_data.head()


# # Splitting the data for training

# In[39]:


X=car_data.drop(['Car_Name','Selling_Price'],axis=1)
Y=car_data['Selling_Price']


# In[48]:


print('\nX--',X.head(3))
print('\nY--',Y.head(3))


# In[50]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state=2)


# Model Selection

# In[52]:


reg=LinearRegression()
reg.fit(X_train,Y_train)


# In[53]:


# prediction on Training data
training_data_prediction = reg.predict(X_train)


# In[54]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[55]:


plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[57]:


# prediction on Test data
test_data_prediction = reg.predict(X_test)


# In[58]:


# R squared Error
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R squared Error : ", error_score)


# In[59]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


#  # Mean Squared Error

# In[61]:


y_pred = reg.predict(X_test)


# In[66]:


mse = mean_squared_error(Y_test, y_pred)
mse


# # Random Forest Algo

# In[67]:


from sklearn.ensemble import RandomForestRegressor


# In[68]:


model = RandomForestRegressor(n_estimators=100, random_state=42)


# In[69]:


model.fit(X_train,Y_train)


# In[70]:


training_data_prediction = model.predict(X_train)


# In[71]:


# R squared Error
error_score = metrics.r2_score(Y_train, training_data_prediction)
print("R squared Error : ", error_score)


# In[72]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[78]:


y_pred1 = reg.predict(X_test)
mse = mean_squared_error(Y_test, y_pred1)
mse


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




