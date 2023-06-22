#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[4]:


df=pd.read_csv('car data.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df.describe().T


# In[8]:


# Unique Data od catagorical Data

print(df.Fuel_Type.unique())
print(df.Seller_Type.unique())
print(df.Transmission.unique())
print(df.Owner.unique())


# In[9]:


#Value Counts of Caragorical Data

print("Fuel Type\n",df.Fuel_Type.value_counts())
print("\nSeller Type \n",df.Seller_Type.value_counts())
print("\nTransmission \n",df.Transmission.value_counts())
print("\nOwener \n",df.Owner.value_counts())


# In[10]:


# Create a figure and subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

# Plot countplot in the first subplot
sns.countplot(data=df, x='Fuel_Type', ax=axs[0, 0])

# Plot countplot in the second subplot
sns.countplot(data=df, x='Seller_Type', ax=axs[0, 1])

# Plot countplot in the third subplot
sns.countplot(data=df, x='Transmission', ax=axs[1, 0])

# Plot countplot in the fourth subplot
sns.countplot(data=df, x='Owner', ax=axs[1, 1])

# Add titles and labels for each subplot
axs[0, 0].set_title('Fuel Type')
axs[0, 0].set_xlabel('Categories')
axs[0, 0].set_ylabel('Count')

axs[0, 1].set_title('Seller Type')
axs[0, 1].set_xlabel('Categories')
axs[0, 1].set_ylabel('Count')

axs[1, 0].set_title('Transmission')
axs[1, 0].set_xlabel('Categories')
axs[1, 0].set_ylabel('Count')

axs[1, 1].set_title('Owner')
axs[1, 1].set_xlabel('Categories')
axs[1, 1].set_ylabel('Count')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plot
plt.show()


# In[11]:


df.columns


# In[12]:


# Car name is not a factor for Machine learning
car_data=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
car_data.head()


# In[13]:


# To findout how old the car is by Year
car_data['Current_year']=2023
car_data.head()


# In[14]:


car_data['How_old']=car_data['Current_year']-car_data['Year']
car_data.head()


# In[15]:


car_data.drop(['Year'],axis=1,inplace=True)


# In[16]:


car_data.head()


# In[17]:


car_data.drop(['Current_year'],axis=1,inplace=True)


# In[18]:


car_data.head()


# Visualizing the correlation between features

# In[98]:


corrmat = car_data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,10))
#plot heat map
g=sns.heatmap(car_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# Dummy Variable Trap
# 
# The "dummy variable trap" refers to a situation where there is perfect multicollinearity among the independent variables in a regression model due to the inclusion of a set of dummy variables. This occurs when one or more dummy variables can be predicted perfectly using the remaining dummy variables in the model.

# In[19]:


# To get the dummy data prevention trap  # dummy variable trap
car_data=pd.get_dummies(car_data,drop_first=True)


# In[20]:


car_data.head()  


# In[21]:


car_data.corr()


# In[22]:


# sns.pairplot(car_data) 
# by doing this we cannot get much info 


# In[64]:


corrmat=car_data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))


# In[ ]:





# In[37]:


#independent Feature
X=car_data[['Present_Price', 'Kms_Driven', 'Owner', 'How_old',                             
       'Fuel_Type_Diesel', 'Fuel_Type_Petrol', 'Seller_Type_Individual',
       'Transmission_Manual']]
#or X=car_data.iloc[:,1:]

#Dependent Feature
y=car_data[["Selling_Price"]]
#or Y=car_data.iloc[:,0]


# In[38]:


print(X.head())
print(y.head())


# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=40)


# In[42]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # RandomForenstRegressor

# In[43]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()  #shift


# In[44]:


rf_random.fit(X_train,y_train)


# In[47]:


y_rf_random_pred = rf_random.predict(X_test)


# In[50]:


#Evaluating The Algorithm
from sklearn import metrics
score1 = metrics.r2_score(y_test,y_rf_random_pred)
score1


# # LineraRegression

# In[53]:


from sklearn.linear_model import LinearRegression


# In[54]:


lir=LinearRegression()


# In[55]:


lir.fit(X_train,y_train)


# In[57]:


y_lir=lir.predict(X_test)


# In[58]:


#Evaluating The Algorithm
score2=metrics.r2_score(y_test,y_lir)
score2


# # GradientBoostingRegressor

# In[59]:


from sklearn.ensemble import GradientBoostingRegressor


# In[60]:


gbr=GradientBoostingRegressor()


# In[61]:


gbr.fit(X_train,y_train)


# In[62]:


y_gbr_pred=gbr.predict(X_test)


# In[63]:


score3=metrics.r2_score(y_test,y_gbr_pred)
score3


# In[65]:


compare_model=pd.DataFrame({'Models':['RandomForenstRegressor','LineraRegression','GradientBoostingRegressor']
                            ,'r2_score':[score1,score2,score3]})
compare_model


# In[87]:


# Comparing model accuracy
plt.figure(figsize=(7,7))
plt.bar(compare_model.Models,compare_model.r2_score,color='orange')


# GradientBoostingRegressor has the  highest accuracy of 0.980264

# In[91]:


## Mean and R squared Errors For GradientBoostingRegressor

print("MAE : ", metrics.mean_absolute_error(y_test, y_gbr_pred))
print("MSE : ", metrics.mean_squared_error(y_test, y_gbr_pred))
print("RMSE : ", np.sqrt(metrics.mean_squared_error(y_test, y_gbr_pred)))
print("R squared Error : ", metrics.r2_score(y_test, y_gbr_pred))


# Hence, GradientBoostingRegressor has the most accuracy among all three model have used.

# In[94]:


y_train_pred = gbr.predict(X_train)


# In[95]:


plt.scatter(y_train, y_train_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(" Actual Prices vs Predicted Prices")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




