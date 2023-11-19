#!/usr/bin/env python
# coding: utf-8

# 
# 
# FEATURES:<br>
# 
# Brand: The brand and model of the car.
# 
# Location: The location in which the car is being sold or is available for purchase.
# 
# Year: The year or edition of the model.
# 
# Kilometers_Driven: The total kilometres driven in the car by the previous owner(s) in KM.
# 
# Fuel_Type: The type of fuel used by the car.
# 
# Transmission: The type of transmission used by the car.
# 
# Owner_Type: Whether the ownership is Firsthand, Second hand or other.
# 
# Mileage: The standard mileage offered by the car company in kmpl or km/kg
# 
# Engine: The displacement volume of the engine in cc.
# 
# Power: The maximum power of the engine in bhp.
# 
# Seats: The number of seats in the car.
# 
# Price: The price of the used car in INR Lakhs.

# In[1]:


get_ipython().system('pip install xgboost')


# In[2]:


pip install XlsxWriter


# In[3]:


pip install category_encoders


# <B><h3>IMPORTING LIBRARIES</h3></B>

# In[4]:


# importing libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt 
import seaborn as sn                   
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import re
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error as mse
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from math import sqrt
from sklearn import neighbors
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore")


# <br><h2>DATA COLLECTION:</h2></br>
# 

# <h2><I> TRAIN DATA AND TEST DATA</I></H2><BR>

# In[5]:


# Read target variable data
train_y = pd.read_csv('C:\\Users\\Ishanya\\Desktop\\DSML Project\\1\\data\\RAWDATA\\training_data_targets.csv')

# Rename the first column to 'Selling_Price'
train_y = train_y.rename(columns={train_y.columns[0]: 'Selling_Price'})

# Display the first few rows of the modified target variable data
print(train_y.head())


# In[6]:


# Read DataFrames
train_x = pd.read_csv('C:\\Users\\Ishanya\\Desktop\\DSML Project\\1\\data\\RAWDATA\\training_data.csv')
train_y = pd.read_csv('C:\\Users\\Ishanya\\Desktop\\DSML Project\\1\\data\\RAWDATA\\training_data_targets.csv')

# Concatenate DataFrames along columns
train_x['Selling_Price'] = train_y['Selling_Price']



#test data

test_x = pd.read_csv(r"C:\Users\Ishanya\Desktop\DSML Project\1\data\RAWDATA\test_data.csv")
print(test_x)


# In[7]:


print(train_x)


# In[8]:


print(train_x.tail())

# We see that train data and targets are perfectly merged


# In[9]:


print(test_x)


# In[10]:


train_x.info()


# In[11]:


train_x.describe()


# In[12]:


test_x.describe()


# In[13]:


# Converting to Excel for verificatoin
train_x.to_excel('C:\\Users\\Ishanya\\Desktop\\DSML Project\\1\\data\\RAWDATA\\combined_data_altered.xlsx', index=False)


# <b><h1>DATA CLEANING</h1></b>

# <h2>Removing Null Values:</h2>

# In[14]:


train_x.isnull().sum()


# In[15]:


test_x.isnull().sum()


# In[16]:


# Find and replace specific values with NaN for each column in train_x
replacement_dict = {
    '0.0 kmpl': np.NaN,  # Replace '0.0' with NaN in the 'Mileage' column
    'null bhp': np.NaN,  # Replace 'null' with NaN in the 'Power' column (assuming 'null' is a placeholder for missing values)
    0: np.NaN  # Replace 0 with NaN in other numeric columns, adjust this based on your dataset
}

train_x = train_x.replace(replacement_dict)

print(train_x)


# In[17]:


# Find and replace specific values with NaN for each column in test_x
replacement_dict = {
    '0.0 kmpl': np.NaN,  # Replace '0.0' with NaN in the 'Mileage' column
    'null bhp': np.NaN,  # Replace 'null' with NaN in the 'Power' column (assuming 'null' is a placeholder for missing values)
    0: np.NaN  # Replace 0 with NaN in other numeric columns, adjust this based on your dataset
}

test_x = test_x.replace(replacement_dict)

# Display the first 50 rows of the modified DataFrame
print(test_x)


# In[18]:


# Count of 0 values for each feature in train_x
count_of_zerosBrand = (train_x['Brand'] == 0).sum()
count_of_zerosLocation = (train_x['Location'] == 0).sum()
count_of_zerosYear = (train_x['Year'] == 0).sum()
count_of_zerosKilometers_Driven = (train_x['Kilometers_Driven'] == 0).sum()
count_of_zerosFuel_Type = (train_x['Fuel_Type'] == 0).sum()
count_of_zerosTransmission = (train_x['Transmission'] == 0).sum()
count_of_zerosOwner_Type = (train_x['Owner_Type'] == 0).sum()
count_of_zerosMileage = (train_x['Mileage'] == 0).sum()
count_of_zerosEngine = (train_x['Engine'] == 0).sum()
count_of_zerosPower = (train_x['Power'] == 0).sum()
count_of_zerosSeats = (train_x['Seats'] == 0).sum()
count_of_zerosSelling_Price = (train_x['Selling_Price'] == 0).sum()

# Print the counts
print("Count of 0 values in 'Brand' column:", count_of_zerosBrand)
print("Count of 0 values in 'Location' column:", count_of_zerosLocation)
print("Count of 0 values in 'Year' column:", count_of_zerosYear)
print("Count of 0 values in 'Kilometers_Driven' column:", count_of_zerosKilometers_Driven)
print("Count of 0 values in 'Fuel_Type' column:", count_of_zerosFuel_Type)
print("Count of 0 values in 'Transmission' column:", count_of_zerosTransmission)
print("Count of 0 values in 'Owner_Type' column:", count_of_zerosOwner_Type)
print("Count of 0 values in 'Mileage' column:", count_of_zerosMileage)
print("Count of 0 values in 'Engine' column:", count_of_zerosEngine)
print("Count of 0 values in 'Power' column:", count_of_zerosPower)
print("Count of 0 values in 'Seats' column:", count_of_zerosSeats)
print("Count of 0 values in 'Selling_Price' column:", count_of_zerosSelling_Price)


# In[19]:


# Count of 0 values for each feature in test_x
count_of_zerosBrand = (test_x['Brand'] == 0).sum()
count_of_zerosLocation = (test_x['Location'] == 0).sum()
count_of_zerosYear = (test_x['Year'] == 0).sum()
count_of_zerosKilometers_Driven = (test_x['Kilometers_Driven'] == 0).sum()
count_of_zerosFuel_Type = (test_x['Fuel_Type'] == 0).sum()
count_of_zerosTransmission = (test_x['Transmission'] == 0).sum()
count_of_zerosOwner_Type = (test_x['Owner_Type'] == 0).sum()
count_of_zerosMileage = (test_x['Mileage'] == 0).sum()
count_of_zerosEngine = (test_x['Engine'] == 0).sum()
count_of_zerosPower = (test_x['Power'] == 0).sum()
count_of_zerosSeats = (test_x['Seats'] == 0).sum()


# Print the counts
print("Count of 0 values in 'Brand' column:", count_of_zerosBrand)
print("Count of 0 values in 'Location' column:", count_of_zerosLocation)
print("Count of 0 values in 'Year' column:", count_of_zerosYear)
print("Count of 0 values in 'Kilometers_Driven' column:", count_of_zerosKilometers_Driven)
print("Count of 0 values in 'Fuel_Type' column:", count_of_zerosFuel_Type)
print("Count of 0 values in 'Transmission' column:", count_of_zerosTransmission)
print("Count of 0 values in 'Owner_Type' column:", count_of_zerosOwner_Type)
print("Count of 0 values in 'Mileage' column:", count_of_zerosMileage)
print("Count of 0 values in 'Engine' column:", count_of_zerosEngine)
print("Count of 0 values in 'Power' column:", count_of_zerosPower)
print("Count of 0 values in 'Seats' column:", count_of_zerosSeats)


# In[20]:


train_x.isnull().sum()


# In[21]:


test_x.isnull().sum()


# In[22]:


#function for imputing missing values with mode
def impute_with_mode(data, column):
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)
    print(f"Mode of {column}: {mode_value}")

    # Check if there are any missing values left in the column after imputation
    missing_values = data[column].isnull().sum()
    print(f"Missing values in {column} column after imputation: {missing_values}")

# Columns to impute
columns_to_impute = ['Mileage', 'Engine', 'Power', 'Seats']

# Impute missing values for each specified column
for column in columns_to_impute:
    impute_with_mode(train_x, column)


# In[23]:


# function for imputing missing values with mode
def impute_with_mode(data, column):
    mode_value = data[column].mode()[0]
    data[column].fillna(mode_value, inplace=True)
    print(f"Mode of {column}: {mode_value}")

    # Check if there are any missing values left in the column after imputation
    missing_values = data[column].isnull().sum()
    print(f"Missing values in {column} column after imputation: {missing_values}")

# Columns to impute
columns_to_impute = ['Mileage', 'Engine', 'Power', 'Seats']

# Impute missing values for each specified column in the test_x DataFrame
for column in columns_to_impute:
    impute_with_mode(test_x, column)


# In[24]:


# Removing units from Mileage, Engine, and Power columns in the 'train_x' DataFrame
train_x['Mileage'] = train_x['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))
train_x['Mileage'] = train_x['Mileage'].apply(lambda x: str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))

train_x['Engine'] = train_x['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))

train_x['Power'] = train_x['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))


# In[25]:


# Removing units from Mileage, Engine, and Power columns in the 'test_x' DataFrame
test_x['Mileage'] = test_x['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))
test_x['Mileage'] = test_x['Mileage'].apply(lambda x: str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))

test_x['Engine'] = test_x['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))

test_x['Power'] = test_x['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))


# In[26]:


#converting into float
train_x['Mileage']=train_x['Mileage'].astype('float')

#converting into float
train_x['Engine']=train_x['Engine'].astype('float')

#converting into float
train_x['Power']=train_x['Power'].astype('float')

print(train_x.dtypes)


# In[27]:


# Converting 'Mileage' to float
test_x['Mileage'] = test_x['Mileage'].astype('float')

# Converting 'Engine' to float
test_x['Engine'] = test_x['Engine'].astype('float')

# Converting 'Power' to float
test_x['Power'] = test_x['Power'].astype('float')

# Display the updated data types
print(test_x.dtypes)


# In[28]:


#Spliting Brand1 into Brand,Model and Version 
train_x['Brand1']=train_x['Brand'].str.split(' ').str[0]
train_x['Model']=train_x['Brand'].str.split(' ').str[1]
train_x['Version']=train_x['Brand'].str.split(' ').str[2:7].str.join(" ") 
print(train_x)


# In[29]:


test_x['Brand1'] = test_x['Brand'].str.split(' ').str[0]
test_x['Model'] = test_x['Brand'].str.split(' ').str[1]
test_x['Version'] = test_x['Brand'].str.split(' ').str[2:7].str.join(" ")
print(test_x)


# In[30]:


unique_counts = train_x.nunique()
print(unique_counts)



# In[31]:


# Check unique values in the 'Mileage' column
unique_mileage_values_test = test_x['Mileage'].unique()
print("Unique values in 'Mileage' (test_x):", unique_mileage_values_test)

# Check unique values in the 'Engine' column
unique_engine_values_test = test_x['Engine'].unique()
print("Unique values in 'Engine' (test_x):", unique_engine_values_test)

# Check unique values in the 'Power' column
unique_power_values_test = test_x['Power'].unique()
print("Unique values in 'Power' (test_x):", unique_power_values_test)

# Check unique values in the 'Seats' column
unique_seats_values_test = test_x['Seats'].unique()
print("Unique values in 'Seats' (test_x):", unique_seats_values_test)


# In[32]:


train=train_x
print(train)


# In[33]:


test=test_x
print(test)


# In[34]:


unique_Brand1_values = train['Brand1'].unique()
print("Unique values in 'Brand1':", unique_Brand1_values)


# In[35]:


#merging Isuzu and ISUZU
train.Brand1[train.Brand1=='Isuzu']='ISUZU'


# In[36]:


# Get the list of column names in test_x
test_columns = test_x.columns

# Check if unique categories in each column of test_x are a subset of train_x
for column_name in test_columns:
    is_subset = set(test_x[column_name].unique()).issubset(set(train_x[column_name].unique()))
    
    if is_subset:
        print(f"Categories in {column_name} of test_x are a subset of train_x.")
    else:
        print(f"Categories in {column_name} of test_x are not a subset of train_x.")


# In[37]:


# Convert brand names to lowercase for consistency
unique_values_train = train['Brand1'].str.lower().unique()
unique_values_test = test['Brand1'].str.lower().unique()

# Find the brand names that are not common
not_common_brands_train = set(unique_values_train) - set(unique_values_test)
not_common_brands_test = set(unique_values_test) - set(unique_values_train)

print("Brand names not common in the training dataset:")
print(not_common_brands_train)

print("\nBrand names not common in the test dataset:")
print(not_common_brands_test)

# Count occurrences of not common brand names in the training dataset
count_not_common_train = train['Brand1'].str.lower().isin(not_common_brands_train).sum()
print("\nCount of occurrences in the training dataset:", count_not_common_train)

# Count occurrences of not common brand names in the test dataset
count_not_common_test = test['Brand1'].str.lower().isin(not_common_brands_test).sum()
print("Count of occurrences in the test dataset:", count_not_common_test)


# In[38]:


# Convert brand names to lowercase for consistency
train['Brand1'] = train['Brand1'].str.lower()
test['Brand1'] = test['Brand1'].str.lower()

# Find the common brand names
common_brands = set(train['Brand1'].unique()).intersection(set(test['Brand1'].unique()))

# Convert common brands to lowercase
common_brands_lower = set(brand.lower() for brand in common_brands)

# Drop rows in the test dataset where 'Brand1' is 'ambassador' or 'smart'
test.drop(test[test['Brand1'].str.lower().isin(['ambassador', 'smart'])].index, inplace=True)

# Update the count after removal
count_after_removal_test = test.shape[0]

print("\nCount of rows after removal in the test dataset:", count_after_removal_test)


# In[39]:


train['Age']=2020-train['Year']


# In[40]:


test['Age']=2020-test['Year']


# In[41]:


train


# In[42]:


test


# <br><H1> EXPLORATORY DATA ANALYSIS AND DATA VISUALIZATION:</H1><br>

# In[43]:


import seaborn as sns

# Plotting the distribution of 'Price' in the 'train' dataset
sns.distplot(train['Selling_Price'])

# Printing skewness and kurtosis
print("Skewness: %f" % train['Selling_Price'].skew())
print("Kurtosis: %f" % train['Selling_Price'].kurt())

#Positive skewness (3.38) indicates a right-leaning distribution for 'Selling Price,' suggesting a concentration of lower-priced cars with a few significantly higher-priced outliers.


# In[44]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot 1: Location
train['Location'].value_counts().plot(kind='bar', ax=axs[0, 0])
axs[0, 0].set_title('Location')

# Plot 2: Fuel_Type
train['Fuel_Type'].value_counts().plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title('Fuel_Type')

# Plot 3: Owner_Type
train['Owner_Type'].value_counts().plot(kind='bar', ax=axs[1, 0])
axs[1, 0].set_title('Owner_Type')

# Plot 4: Year
train['Year'].value_counts().plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title('Year')

# Plot 5: Brand1
train['Brand1'].value_counts().plot(kind='bar', ax=axs[2, 0])
axs[2, 0].set_title('Brand1')

# Plot 6: Age
train['Age'].value_counts().sort_index().plot(kind='bar', ax=axs[2, 1])
axs[2, 1].set_title('Age')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()


# In[45]:


import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

# Plot 1: Location
test['Location'].value_counts().plot(kind='bar', ax=axs[0, 0])
axs[0, 0].set_title('Location')

# Plot 2: Fuel_Type
test['Fuel_Type'].value_counts().plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title('Fuel_Type')

# Plot 3: Owner_Type
test['Owner_Type'].value_counts().plot(kind='bar', ax=axs[1, 0])
axs[1, 0].set_title('Owner_Type')

# Plot 4: Year
test['Year'].value_counts().plot(kind='bar', ax=axs[1, 1])
axs[1, 1].set_title('Year')

# Plot 5: Brand1
test['Brand1'].value_counts().plot(kind='bar', ax=axs[2, 0])
axs[2, 0].set_title('Brand1')

# Plot 6: Age
test['Age'].value_counts().sort_index().plot(kind='bar', ax=axs[2, 1])
axs[2, 1].set_title('Age')

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Selling_Price' is the column you want to visualize
xprops = ['Location', 'Year', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Seats']
yprop = 'Selling_Price'

plt.figure(figsize=(20, 15))

for i, xprop in enumerate(xprops, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(data=train, x=xprop, y=yprop)
    plt.xlabel('{} range'.format(xprop), size=14)
    plt.ylabel('Number of {}'.format(yprop), size=14)
    plt.title('Boxplot of {} vs {}'.format(xprop, yprop), size=16)

plt.tight_layout()
plt.show()



# In[47]:


plt.figure(figsize=(20,5))
sn.scatterplot(x='Kilometers_Driven',y='Selling_Price',data=train, hue='Fuel_Type')


# In[48]:


plt.figure(figsize=(20,5))
sn.scatterplot(x='Kilometers_Driven',y='Selling_Price',data=train, hue='Location')


# In[49]:


plt.figure(figsize=(20,5))
sn.scatterplot(x='Mileage',y='Selling_Price',data=train, hue='Brand1')


# In[50]:


sn.distplot(train["Year"])


# In[51]:


sn.pairplot(train)


# In[52]:


sn.pairplot(test)


# <H1>Checking for outliers:</H1>

# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Boxplot for Kilometers_Driven
sns.boxplot(data=train, x='Kilometers_Driven', ax=axes[0, 0])
axes[0, 0].set_title('Kilometers Driven')

# Boxplot for Mileage
sns.boxplot(data=train, x='Mileage', ax=axes[0, 1])
axes[0, 1].set_title('Mileage')

# Boxplot for Engine
sns.boxplot(data=train, x='Engine', ax=axes[1, 0])
axes[1, 0].set_title('Engine')

# Boxplot for Power
sns.boxplot(data=train, x='Power', ax=axes[1, 1])
axes[1, 1].set_title('Power')

# Adjust layout
plt.tight_layout()
plt.show()


# <h1>SCALING AND HANDLING OUTLIERS WITH ROBUST SCALAR:</H1>

# In[54]:


from sklearn.preprocessing import RobustScaler

# Selecting only the numeric columns
numeric_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power']

# Creating a RobustScaler instance
scaler = RobustScaler()

# Fit and transform the selected numeric columns
train[numeric_columns] = scaler.fit_transform(train[numeric_columns])

# Display the DataFrame after scaling
print(train.head())


# In[55]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Boxplot for Kilometers_Driven
sns.boxplot(data=train, x='Kilometers_Driven', ax=axes[0, 0])
axes[0, 0].set_title('Kilometers Driven')

# Boxplot for Mileage
sns.boxplot(data=train, x='Mileage', ax=axes[0, 1])
axes[0, 1].set_title('Mileage')

# Boxplot for Engine
sns.boxplot(data=train, x='Engine', ax=axes[1, 0])
axes[1, 0].set_title('Engine')

# Boxplot for Power
sns.boxplot(data=train, x='Power', ax=axes[1, 1])
axes[1, 1].set_title('Power')

# Adjust layout
plt.tight_layout()
plt.show()


# In[56]:


train


# <H5>TEST DATA </H5>

# In[57]:


# BEFORE SCALING

import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Boxplot for Kilometers_Driven
sns.boxplot(data=test_x, x='Kilometers_Driven', ax=axes[0, 0])
axes[0, 0].set_title('Kilometers Driven')

# Boxplot for Mileage
sns.boxplot(data=test_x, x='Mileage', ax=axes[0, 1])
axes[0, 1].set_title('Mileage')

# Boxplot for Engine
sns.boxplot(data=test_x, x='Engine', ax=axes[1, 0])
axes[1, 0].set_title('Engine')

# Boxplot for Power
sns.boxplot(data=test_x, x='Power', ax=axes[1, 1])
axes[1, 1].set_title('Power')

# Adjust layout
plt.tight_layout()
plt.show()


# In[58]:


from sklearn.preprocessing import RobustScaler


# Selecting only the numeric columns
numeric_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power']

# Creating a RobustScaler instance
scaler = RobustScaler()

# Fit and transform the selected numeric columns
test_x[numeric_columns] = scaler.fit_transform(test_x[numeric_columns])

# Display the DataFrame after scaling
print(test_x.head())


# In[59]:


# AFTER SCALING


import matplotlib.pyplot as plt
import seaborn as sns

# Set up the subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Boxplot for Kilometers_Driven
sns.boxplot(data=test_x, x='Kilometers_Driven', ax=axes[0, 0])
axes[0, 0].set_title('Kilometers Driven')

# Boxplot for Mileage
sns.boxplot(data=test_x, x='Mileage', ax=axes[0, 1])
axes[0, 1].set_title('Mileage')

# Boxplot for Engine
sns.boxplot(data=test_x, x='Engine', ax=axes[1, 0])
axes[1, 0].set_title('Engine')

# Boxplot for Power
sns.boxplot(data=test_x, x='Power', ax=axes[1, 1])
axes[1, 1].set_title('Power')

# Adjust layout
plt.tight_layout()
plt.show()


# In[60]:


# Exclude non-numeric columns from the DataFrame
numeric_train = train.select_dtypes(include=[np.number])

# Calculate the correlation matrix for the numeric columns
corr = numeric_train.corr()

# Create a heatmap of the correlation matrix
sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sn.diverging_palette(220, 20, as_cmap=True))


# In[61]:


pd.crosstab(train.Location, train.Owner_Type)


# In[62]:


train.columns


# In[63]:


pd.crosstab(train.Brand1,train.Location)


# In[64]:


pd.crosstab(train.Brand1,train.Fuel_Type)


# In[65]:


print(train)


# In[66]:


train


# In[67]:


test


# In[68]:


train.describe()


# In[69]:


test.describe()


# <h1> ONE HOT ENCODING </h1>

# In[70]:


train.dtypes


# In[71]:


test.dtypes


# In[72]:


#Creating dummy dataframe to create map of all possible unique column names provided in train and test files


# In[73]:


train.head


# In[74]:


test.head()


# In[75]:


# List all the column names
columns = train.columns
print(columns)


# In[76]:


# List all the column names
columns = test.columns
print(columns)


# In[77]:


#import pandas as pd

# Perform one-hot encoding on categorical columns in the train set
train_encoded = pd.get_dummies(train, columns=['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand1'])

# Perform one-hot encoding on categorical columns in the test set
test_encoded = pd.get_dummies(test, columns=['Location', 'Fuel_Type', 'Transmission', 'Owner_Type', 'Brand1'])

# Display the first few rows of the encoded train dataset
print(train_encoded.head())

# Display the first few rows of the encoded test dataset
print(test_encoded.head())


# In[78]:


# Select only numeric columns for correlation analysis
numeric_columns = train_encoded.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = train_encoded[numeric_columns].corr()

# Sort and get the correlation with the target variable
correlation_with_target = correlation_matrix['Selling_Price'].abs().sort_values(ascending=False)
selected_features_corr = correlation_with_target[1:].index

# Print correlation values
print("Correlation with Selling_Price:")
print(correlation_with_target)


# In[79]:


print(train)


# In[80]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt



df3_inputs = train.drop('Selling_Price', axis=1)
df3_target = train['Selling_Price']

# Identify categorical columns
categorical_columns = df3_inputs.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
df3_inputs_encoded = pd.DataFrame(onehot_encoder.fit_transform(df3_inputs[categorical_columns]))

# Get the feature names after one-hot encoding
feature_names = onehot_encoder.get_feature_names_out(input_features=categorical_columns)
df3_inputs_encoded.columns = feature_names

# Concatenate the one-hot encoded features with the remaining numeric features
df3_inputs = pd.concat([df3_inputs.drop(columns=categorical_columns), df3_inputs_encoded], axis=1)

# Initialize the ExtraTreesRegressor
model = ExtraTreesRegressor()

# Fit the model on the training data
model.fit(df3_inputs, df3_target)

# Use inbuilt class feature_importances of ExtraTreeRegressor
feat_importances = pd.Series(model.feature_importances_, index=df3_inputs.columns)

# Plot graph of feature importances for better visualization
plt.figure(figsize=(11, 5))
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 5 Features")
feat_importances.nlargest(5).plot(kind='barh', color='#D98880')  # You can change the color as desired
plt.grid(color='black', linestyle='-.', linewidth=0.7)
plt.show()


# In[81]:


#FINAL ENCODING USING ONE HOT ENCODING ON 'Location','Fuel_Type', 'Transmission', 'Owner_Type' AS BRAND1 HAS NO SIGNIFICANCE

# Specify the categorical columns for one-hot encoding based on top features
columns_to_encode = ['Location','Fuel_Type', 'Transmission', 'Owner_Type']

# Perform one-hot encoding on the training data with 1s and 0s
train_encoded = pd.get_dummies(train, columns=columns_to_encode)

# Perform one-hot encoding on the test data with 1s and 0s
test_encoded = pd.get_dummies(test, columns=columns_to_encode)


# In[82]:


train_encoded.columns


# In[83]:


test_encoded.columns


# In[84]:


train_copy=train_encoded
train_copy


# In[85]:


test_copy=test_encoded
test_copy


# In[86]:


test_copy.columns


# In[87]:


#del test_copy['Version']


# In[88]:


#del test_copy['Brand']


# In[89]:


#del test_copy['Brand1']


# In[90]:


#del test_copy['Year']


# In[91]:


#del test_copy['Model']


# In[92]:


train_copy.dtypes


# In[93]:


test_copy.dtypes


# In[94]:


#Splitting data into independent and dependent data
 
X=train_copy[['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
       'Age', 'Location_Ahmedabad', 'Location_Bangalore',
       'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
       'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
       'Location_Kolkata', 'Location_Mumbai', 'Location_Pune', 'Fuel_Type_CNG',
       'Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_LPG',
       'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual',
       'Owner_Type_First', 'Owner_Type_Fourth & Above', 'Owner_Type_Second',
       'Owner_Type_Third']]
y=train_copy.Selling_Price


# In[95]:


X


# In[96]:


X.columns


# In[97]:


test_data=test_copy[['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
       'Age', 'Location_Ahmedabad', 'Location_Bangalore', 'Location_Chennai',
       'Location_Coimbatore', 'Location_Delhi', 'Location_Hyderabad',
       'Location_Jaipur', 'Location_Kochi', 'Location_Kolkata',
       'Location_Mumbai', 'Location_Pune', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
       'Fuel_Type_Electric', 'Fuel_Type_LPG', 'Fuel_Type_Petrol',
       'Transmission_Automatic', 'Transmission_Manual', 'Owner_Type_First',
       'Owner_Type_Fourth & Above', 'Owner_Type_Second', 'Owner_Type_Third']]


# In[98]:


test_data


# <h2> CROSS VALIDATION </H2>

# <h1> TRAINING THE DATA USING VARIOUS REGRESSION MODELS </h1>

# In[99]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Assuming 'X' and 'y' are your features and target variable
X = train_copy[['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
                'Age', 'Location_Ahmedabad', 'Location_Bangalore',
                'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
                'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
                'Location_Kolkata', 'Location_Mumbai', 'Location_Pune', 'Fuel_Type_CNG',
                'Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_LPG',
                'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual',
                'Owner_Type_First', 'Owner_Type_Fourth & Above', 'Owner_Type_Second',
                'Owner_Type_Third']]
y = train_copy.Selling_Price

# Splitting the data into training, testing, and validation sets
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=123)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)

# Feature Selection
selector_mutual_info = SelectKBest(score_func=mutual_info_regression, k=10)  # You can adjust 'k' as needed

# Create pipelines for each model with scaling and feature selection
linreg_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', LinearRegression())
])

lasso_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', Lasso())
])

ridge_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', Ridge())
])

dt_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', DecisionTreeRegressor())
])

rf_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', RandomForestRegressor())
])

gbr_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', GradientBoostingRegressor())
])

xgb_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', XGBRegressor())
])

# Define default parameters for each model
default_params = {
    'Linear Regression': {
        'selector__k': [5, 10, 15],
        'model__fit_intercept': [True, False]
    },
    'Lasso Regression': {
        'selector__k': [5, 10, 15],
        'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    },
    'Ridge Regression': {
        'selector__k': [5, 10, 15],
        'model__alpha': [0.1, 1, 10, 100]
    },
    'Decision Tree': {
        'selector__k': [5, 10, 15],
        'model__criterion': ['mse', 'friedman_mse', 'mae'],
        'model__max_features': ['auto', 'sqrt', 'log2'],
        'model__max_depth': [10, 20, 30, 40],
        'model__ccp_alpha': [0.001, 0.01, 0.1]
    },
    'Random Forest': {
        'selector__k': [5, 10, 15],
        'model__n_estimators': [50, 100, 200, 300],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'selector__k': [5, 10, 15],
        'model__n_estimators': [50, 100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6]
    },
    'XGBoost': {
        'selector__k': [5, 10, 15],
        'model__n_estimators': [50, 100, 200, 300],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5, 6]
    },
}

# Create a dictionary of pipelines and parameter grids
pipelines = {
    'Linear Regression': (linreg_pipe, default_params['Linear Regression']),
    'Lasso Regression': (lasso_pipe, default_params['Lasso Regression']),
    'Ridge Regression': (ridge_pipe, default_params['Ridge Regression']),
    'Decision Tree': (dt_pipe, default_params['Decision Tree']),
    'Random Forest': (rf_pipe, default_params['Random Forest']),
    'Gradient Boosting': (gbr_pipe, default_params['Gradient Boosting']),
    'XGBoost': (xgb_pipe, default_params['XGBoost']),
}

# Fit models using GridSearchCV and print results
results = {}
for model_name, (pipeline, param_grid) in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_temp, y_train_temp)  # Fit the grid search
    y_pred = cross_val_predict(grid_search.best_estimator_, X_train_temp, y_train_temp, cv=5)  # Cross-validation prediction
    mse = mean_squared_error(y_train_temp, y_pred)
    mae = mean_absolute_error(y_train_temp, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train_temp, y_pred)
    
    results[model_name] = {
        'Best Parameters': grid_search.best_estimator_.named_steps['model'].get_params(),
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae,
        'Root Mean Squared Error': rmse,
        'R^2 Score': r2
    }

    # Plotting density vs. residuals and actual vs. predicted in one row
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    residuals = y_train_temp - y_pred
    sns.histplot(residuals, kde=True)
    plt.title(f'Density vs. Residuals - {model_name}')
    plt.xlabel('Residuals')
    plt.ylabel('Density')

    plt.subplot(1, 2, 2)
    plt.scatter(y_train_temp, y_pred)
    plt.title(f'Actual vs. Predicted - {model_name}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()

# Print results
for model_name, result in results.items():
    print(f"{model_name}:\n{result}\n")



# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

# Assuming 'X' and 'y' are your features and target variable
X = train_copy[['Kilometers_Driven', 'Mileage', 'Engine', 'Power', 'Seats',
                'Age', 'Location_Ahmedabad', 'Location_Bangalore',
                'Location_Chennai', 'Location_Coimbatore', 'Location_Delhi',
                'Location_Hyderabad', 'Location_Jaipur', 'Location_Kochi',
                'Location_Kolkata', 'Location_Mumbai', 'Location_Pune', 'Fuel_Type_CNG',
                'Fuel_Type_Diesel', 'Fuel_Type_Electric', 'Fuel_Type_LPG',
                'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual',
                'Owner_Type_First', 'Owner_Type_Fourth & Above', 'Owner_Type_Second',
                'Owner_Type_Third']]
y = train_copy.Selling_Price

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Feature Selection
selector_mutual_info = SelectKBest(score_func=mutual_info_regression, k=10)  # You can adjust 'k' as needed

# Define pipelines
dt_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', DecisionTreeRegressor())
])

ada_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', AdaBoostRegressor())
])

svr_pipe = Pipeline([
    ('selector', selector_mutual_info),
    ('model', SVR())
])

# Define parameter distributions for RandomizedSearchCV
param_dist_dt = {
    'selector__k': [5, 10, 15],
    'model__criterion': ['mse', 'friedman_mse', 'mae'],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth': [10, 20, 30, 40],
    'model__ccp_alpha': [0.001, 0.01, 0.1]
}

param_dist_adaboost = {
    'selector__k': [5, 10, 15],
    'model__base_estimator': [DecisionTreeRegressor(), LinearRegression()],
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': np.linspace(0.01, 0.2, 10)
}

param_dist_svr = {
    'selector__k': [5, 10, 15],
    'model__C': [0.1, 1, 10, 100],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto']
}

# Perform RandomizedSearchCV for Decision Tree
random_search_dt = RandomizedSearchCV(dt_pipe, param_distributions=param_dist_dt, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search_dt.fit(X_train, y_train)

# Perform RandomizedSearchCV for AdaBoost
random_search_adaboost = RandomizedSearchCV(ada_pipe, param_distributions=param_dist_adaboost, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search_adaboost.fit(X_train, y_train)

# Perform RandomizedSearchCV for SVR
random_search_svr = RandomizedSearchCV(svr_pipe, param_distributions=param_dist_svr, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search_svr.fit(X_train, y_train)

# Get best parameters and evaluate the models
best_params_dt = random_search_dt.best_params_
y_pred_dt = cross_val_predict(random_search_dt.best_estimator_, X_test, y_test, cv=5)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
r2_dt = r2_score(y_test, y_pred_dt)

best_params_adaboost = random_search_adaboost.best_params_
y_pred_adaboost = cross_val_predict(random_search_adaboost.best_estimator_, X_test, y_test, cv=5)
mse_adaboost = mean_squared_error(y_test, y_pred_adaboost)
mae_adaboost = mean_absolute_error(y_test, y_pred_adaboost)
rmse_adaboost = np.sqrt(mse_adaboost)
r2_adaboost = r2_score(y_test, y_pred_adaboost)

best_params_svr = random_search_svr.best_params_
y_pred_svr = cross_val_predict(random_search_svr.best_estimator_, X_test, y_test, cv=5)
mse_svr = mean_squared_error(y_test, y_pred_svr)
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mse_svr)
r2_svr = r2_score(y_test, y_pred_svr)

# Print results
print("Decision Tree Results:")
print("Best Parameters:", best_params_dt)
print("Mean Squared Error:", mse_dt)
print("Mean Absolute Error:", mae_dt)
print("Root Mean Squared Error:", rmse_dt)
print("R^2 Score:", r2_dt)

print("\nAdaBoost Results:")
print("Best Parameters:", best_params_adaboost)
print("Mean Squared Error:", mse_adaboost)
print("Mean Absolute Error:", mae_adaboost)
print("Root Mean Squared Error:", rmse_adaboost)
print("R^2 Score:", r2_adaboost)

print("\nSVR Results:")
print("Best Parameters:", best_params_svr)
print("Mean Squared Error:", mse_svr)
print("Mean Absolute Error:", mae_svr)
print("Root Mean Squared Error:", rmse_svr)
print("R^2 Score:", r2_svr)


# In[ ]:


test_copy.dtypes


# In[ ]:


test_copy


# In[ ]:


X.dtypes


# In[ ]:


test_copy.columns


# In[ ]:


train_copy.columns


# In[ ]:


X.columns


# <h1>FINAL PREDICTION<h1> <h2>USIING XGB, RF AND GB MODLE</h2>

# In[ ]:


import os
import pandas as pd

ordered_columns = X_train_temp.columns

# Reorder the columns in test_copy
test_copy_ordered = test_copy[ordered_columns]

# Prediction on the test set of the provided 'test_copy' for XGBoost
prediction_xgb = grid_search.best_estimator_.predict(test_copy_ordered)
sol_xgb = pd.DataFrame({'Selling_Price_XGB': prediction_xgb})

# Save predictions to a text file for XGBoost
folder_path = r'C:\Users\Ishanya\Desktop\DSML Project\1\data\RAWDATA'
sol_xgb.to_csv(os.path.join(folder_path, 'Ishanya_predictions_xgb.txt'), index=False, header=True, sep='\t')

# Save predictions to an Excel file for XGBoost
sol_xgb.to_excel(os.path.join(folder_path, 'Ishanya_predictions_xgb.xlsx'), index=False)


# In[ ]:


# Assuming X_train contains the training features used for training the Random Forest model
ordered_columns_rf = X_train_temp.columns

# Reorder the columns in test_copy
test_copy_ordered_rf = test_copy[ordered_columns_rf]

# Prediction on the test set of the provided 'test_copy' for Random Forest
prediction_rf = grid_search.best_estimator_.predict(test_copy_ordered_rf)
sol_rf = pd.DataFrame({'Selling_Price_RF': prediction_rf})

# Save predictions to a text file for Random Forest
folder_path_rf = r'C:\Users\Ishanya\Desktop\DSML Project\1\data\RAWDATA'
sol_rf.to_csv(os.path.join(folder_path_rf, 'Ishanya_predictions_rf.txt'), index=False, header=True, sep='\t')

# Save predictions to an Excel file for Random Forest
sol_rf.to_excel(os.path.join(folder_path_rf, 'Ishanya_predictions_rf.xlsx'), index=False)


# In[ ]:


# Assuming X_train contains the training features used for training the Gradient Boosting model
ordered_columns_gbr = X_train_temp.columns

# Reorder the columns in test_copy
test_copy_ordered_gbr = test_copy[ordered_columns_gbr]

# Prediction on the test set of the provided 'test_copy' for Gradient Boosting
prediction_gbr = grid_search.best_estimator_.predict(test_copy_ordered_gbr)
sol_gbr = pd.DataFrame({'Selling_Price_GBR': prediction_gbr})

# Save predictions to a text file for Gradient Boosting
folder_path_gbr = r'C:\Users\Ishanya\Desktop\DSML Project\1\data\RAWDATA'
sol_gbr.to_csv(os.path.join(folder_path_gbr, 'Ishanya_predictions_gbr.txt'), index=False, header=True, sep='\t')

# Save predictions to an Excel file for Gradient Boosting
sol_gbr.to_excel(os.path.join(folder_path_gbr, 'Ishanya_predictions_gbr.xlsx'), index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




