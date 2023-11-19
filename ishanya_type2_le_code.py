# -*- coding: utf-8 -*-
"""Ishanya_TYPE2_LE_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sM1L9dI_Q6IGs4Kb_wHVQhP4Md4HVq9j

FEATURES:<br>

Brand: The brand and model of the car.

Location: The location in which the car is being sold or is available for purchase.

Year: The year or edition of the model.

Kilometers_Driven: The total kilometres driven in the car by the previous owner(s) in KM.

Fuel_Type: The type of fuel used by the car.

Transmission: The type of transmission used by the car.

Owner_Type: Whether the ownership is Firsthand, Second hand or other.

Mileage: The standard mileage offered by the car company in kmpl or km/kg

Engine: The displacement volume of the engine in cc.

Power: The maximum power of the engine in bhp.

Seats: The number of seats in the car.

Price: The price of the used car in INR Lakhs.
"""

!pip install xgboost

pip install XlsxWriter

"""<B><h3>IMPORTING LIBRARIES</h3></B>"""

# Commented out IPython magic to ensure Python compatibility.
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

# %matplotlib inline
warnings.filterwarnings("ignore")

"""<h2><I> TRAIN DATA</I></H2><BR>"""

# Read target variable data
train_y = pd.read_csv('/content/training_data_targets.csv')

# Rename the first column to 'Selling_Price'
train_y = train_y.rename(columns={train_y.columns[0]: 'Selling_Price'})

# Display the first few rows of the modified target variable data
print(train_y.head())

# Read DataFrames
train_x = pd.read_csv('/content/training_data.csv')
train_y = pd.read_csv('/content/training_data_targets.csv')

# Concatenate DataFrames along columns
train_x['Selling_Price'] = train_y['Selling_Price']

# Display the first few rows of the combined DataFrame
print(train_x.head())

#test data

test_x = pd.read_csv(r"/content/test_data.csv")
print(test_x)

"""<br><h2>DATA COLLECTION:</h2></br>

"""

train_x.describe

train_x

print(train_x.tail())

train_x.info()

train_x.describe()

# Converting to Excel for verificatoin
train_x.to_excel('C:\\Users\\Ishanya\\Desktop\\DSML Project\\1\\data\\RAWDATA\\combined_data_altered.xlsx', index=False)

"""<b><h1>DATA CLEANING</h1></b>

<h2>Removing Null Values:</h2>
"""

train_x.isnull().sum()

# Find and replace specific values with NaN for each column in train_x
replacement_dict = {
    '0.0 kmpl': np.NaN,  # Replace '0.0' with NaN in the 'Mileage' column
    'null bhp': np.NaN,  # Replace 'null' with NaN in the 'Power' column (assuming 'null' is a placeholder for missing values)
    0: np.NaN  # Replace 0 with NaN in other numeric columns, adjust this based on your dataset
}

train_x = train_x.replace(replacement_dict)

# Display the first 50 rows of the modified DataFrame
print(train_x.head(50))

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

train_x.isnull().sum()

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

# Display the first 50 rows of the modified DataFrame
#print(train_x.head(50))

# Removing units from Mileage, Engine, and Power columns in the 'train_x' DataFrame
train_x['Mileage'] = train_x['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))
train_x['Mileage'] = train_x['Mileage'].apply(lambda x: str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))

train_x['Engine'] = train_x['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))

train_x['Power'] = train_x['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))

train_x.isnull().sum()

# Check unique values in the 'Mileage' column
unique_mileage_values = train_x['Mileage'].unique()
print("Unique values in 'Mileage':", unique_mileage_values)

# Check unique values in the 'Engine' column
unique_engine_values = train_x['Engine'].unique()
print("Unique values in 'Engine':", unique_engine_values)

# Check unique values in the 'Power' column
unique_power_values = train_x['Power'].unique()
print("Unique values in 'Power':", unique_power_values)

# Check unique values in the 'Seats' column
unique_seats_values = train_x['Seats'].unique()
print("Unique values in 'Seats':", unique_seats_values)

train_x

train=train_x
train

train.shape

#Spliting Brand1 into Brand,Model and Version
train['Brand1']=train['Brand'].str.split(' ').str[0]
train['Model']=train['Brand'].str.split(' ').str[1]
train['Version']=train['Brand'].str.split(' ').str[2:7].str.join(" ")
train

train.info()

train.shape

#converting into float
train['Mileage']=train['Mileage'].astype('float')

#converting into float
train['Engine']=train['Engine'].astype('float')

#converting into float
train['Power']=train['Power'].astype('float')

print(train.dtypes)

train

train.info()

train.columns

train.info

train.nunique()

train.shape

"""<h5>Checking for outliers:</h5>"""

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

"""<h1>SCALING AND HANDLING OUTLIERS WITH ROBUST SCALAR:</H1>"""

from sklearn.preprocessing import RobustScaler

# Assuming 'train' is your DataFrame

# Selecting only the numeric columns
numeric_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power']

# Creating a RobustScaler instance
scaler = RobustScaler()

# Fit and transform the selected numeric columns
train[numeric_columns] = scaler.fit_transform(train[numeric_columns])

# Display the DataFrame after scaling
print(train.head())

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

train

"""<br><H1> EXPLORATORY DATA ANALYSIS AND DATA VISUALIZATION:</H1><br>"""

train.describe()

# Exclude non-numeric columns from the DataFrame
numeric_train = train.select_dtypes(include=[np.number])

# Calculate the correlation matrix for the numeric columns
corr = numeric_train.corr()

# Create a heatmap of the correlation matrix
sn.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sn.diverging_palette(220, 20, as_cmap=True))

import matplotlib.pyplot as plt

# Assuming N is defined somewhere before this code
N = 10

fig, axs = plt.subplots(3, 2, figsize=(10, 10))

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

# Adjust layout
plt.tight_layout()

# Show the combined plot
plt.show()

pd.crosstab(train.Location, train.Owner_Type)

sn.distplot(train["Year"])

sn.distplot(train["Selling_Price"])

plt.figure(figsize=(20,5))
sn.boxplot(
    data=train,
    x='Fuel_Type',y='Selling_Price',
    )

plt.figure(figsize=(20,5))
sn.boxplot(
    data=train,
    x='Transmission',y='Selling_Price',
   )

plt.figure(figsize=(20,5))
sn.boxplot(
    data=train,
    x='Owner_Type',y='Selling_Price',
    color='red')

plt.figure(figsize=(20,5))
sn.boxplot(
    data=train,
    x='Location',y='Selling_Price',
    color='red')

plt.figure(figsize=(20,5))
sn.scatterplot(x='Kilometers_Driven',y='Selling_Price',data=train, hue='Fuel_Type')

plt.figure(figsize=(20,5))
sn.scatterplot(x='Kilometers_Driven',y='Selling_Price',data=train, hue='Location')

sn.pairplot(train)

plt.figure(figsize=(20,5))
sn.barplot(x="Engine",y="Selling_Price",data=train)

train.columns

plt.figure(figsize=(20,5))
sn.scatterplot(x='Kilometers_Driven',y='Selling_Price',data=train, hue='Brand1')

plt.figure(figsize=(20,5))
sn.scatterplot(x='Mileage',y='Selling_Price',data=train, hue='Brand1')

plt.figure(figsize=(20,5))
sn.scatterplot(x='Engine',y='Selling_Price',data=train, hue='Brand1')

plt.figure(figsize=(20,5))
sn.scatterplot(x='Power',y='Selling_Price',data=train, hue='Brand1')

plt.figure(figsize=(10,11))
sn.boxplot(
    data=train,
    x='Selling_Price',y='Brand1')

pd.crosstab(train.Brand1,train.Location)

pd.crosstab(train.Brand,train.Fuel_Type)

train

"""<H1><i> TEST DATA <i><H1>

<h1>DATA COLLECTION<BR><h1>
"""

test_x = pd.read_csv(r"/content/test_data.csv")
print(test_x.head())

test_x.shape

test_x.describe()

test_x.describe

"""<h1> DATA CLEANING:</H1>

<H2> Removing Null Values </h2>
"""

test_x.isnull().sum()

# Find and replace specific values with NaN for each column in test_x
replacement_dict = {
    '0.0 kmpl': np.NaN,  # Replace '0.0' with NaN in the 'Mileage' column
    'null bhp': np.NaN,  # Replace 'null' with NaN in the 'Power' column (assuming 'null' is a placeholder for missing values)
    0: np.NaN  # Replace 0 with NaN in other numeric columns, adjust this based on your dataset
}

test_x = test_x.replace(replacement_dict)

# Display the first 50 rows of the modified DataFrame
print(test_x.head(50))

test_x.isnull().sum()

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

# Define a function for imputing missing values with mode
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

# Removing units from Mileage, Engine, and Power columns in the 'test_x' DataFrame
test_x['Mileage'] = test_x['Mileage'].apply(lambda x: str(x).replace('kmpl', '') if 'kmpl' in str(x) else str(x))
test_x['Mileage'] = test_x['Mileage'].apply(lambda x: str(x).replace('km/kg', '') if 'km/kg' in str(x) else str(x))

test_x['Engine'] = test_x['Engine'].apply(lambda x: str(x).replace('CC', '') if 'CC' in str(x) else str(x))

test_x['Power'] = test_x['Power'].apply(lambda x: str(x).replace('bhp', '') if 'bhp' in str(x) else str(x))

# Converting 'Mileage' to float
test_x['Mileage'] = test_x['Mileage'].astype('float')

# Converting 'Engine' to float
test_x['Engine'] = test_x['Engine'].astype('float')

# Converting 'Power' to float
test_x['Power'] = test_x['Power'].astype('float')

# Display the updated data types
print(test_x.dtypes)

test_x.isnull().sum()

test_x.columns

test_x.dtypes

# Splitting 'Brand1' into 'Brand', 'Model', and 'Version' in test_x DataFrame
test_x['Brand1'] = test_x['Brand'].str.split(' ').str[0]
test_x['Model'] = test_x['Brand'].str.split(' ').str[1]
test_x['Version'] = test_x['Brand'].str.split(' ').str[2:7].str.join(" ")

test=test_x
test

test

"""<h1>SCALING</h1>"""

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

from sklearn.preprocessing import RobustScaler


# Selecting only the numeric columns
numeric_columns = ['Kilometers_Driven', 'Mileage', 'Engine', 'Power']

# Creating a RobustScaler instance
scaler = RobustScaler()

# Fit and transform the selected numeric columns
test_x[numeric_columns] = scaler.fit_transform(test_x[numeric_columns])

# Display the DataFrame after scaling
print(test_x.head())

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

test

"""<h1> LABEL ENCODING </h1>"""

train.dtypes

test.dtypes

#Creating dummy dataframe to create map of all possible unique column names provided in train and test files

train.head()

test.head()

# List all the column names
columns = train.columns
print(columns)

# List all the column names
columns = test.columns
print(columns)

# Specify the columns to one-hot encode
columns_to_encode = ['Fuel_Type', 'Transmission', 'Owner_Type']

# Perform one-hot encoding on the training data
train_encoded = pd.get_dummies(train, columns=columns_to_encode)

# Perform one-hot encoding on the test data
test_encoded = pd.get_dummies(test, columns=columns_to_encode)

import pandas as pd


# Specify the columns to one-hot encode
columns_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Perform one-hot encoding
df_encoded = pd.get_dummies(train, columns=columns_to_encode, prefix=columns_to_encode)

# Display the first few rows and columns of the encoded DataFrame
print(train_encoded.head())

train_encoded

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

# Assuming train_encoded contains your input features and target variable
# Replace 'Selling_Price' with your actual target variable column name
df3_inputs = train_encoded.drop('Selling_Price', axis=1)
df3_target = train_encoded['Selling_Price']

# Label encode categorical columns
label_encoder = LabelEncoder()
for column in df3_inputs.select_dtypes(include=['object']).columns:
    df3_inputs[column] = label_encoder.fit_transform(df3_inputs[column])

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
plt.title("Top 10 Features Importance")
feat_importances.nlargest(10).plot(kind='barh', color='#D98880')  # You can change the color as desired
plt.grid(color='black', linestyle='-.', linewidth=0.7)
plt.show()

train

# Specify the columns to one-hot encode
columns_to_encode = ['Location', 'Fuel_Type', 'Transmission', 'Owner_Type']

# Perform one-hot encoding on the training data
train_encoded = pd.get_dummies(train, columns=columns_to_encode, drop_first=True)

# Perform one-hot encoding on the test data
test_encoded = pd.get_dummies(test, columns=columns_to_encode)

train_encoded

# Get unique values in the 'Year' column
unique_years = test['Year'].unique()

# Sort the unique values in ascending order
sorted_years = sorted(unique_years)

# To sort in descending order, you can use the reverse parameter
# sorted_years_descending = sorted(unique_years, reverse=True)

# Print the sorted unique years
print(sorted_years)

# Get unique values in the 'Year' column
unique_years1 = train['Year'].unique()

# Sort the unique values in ascending order
sorted_years1 = sorted(unique_years1)


# Print the sorted unique years
print(sorted_years1)

ndf=pd.DataFrame()

ndf['train_n']=" "

ndf['test_n']=" "

ndf.train_n=train.Brand1
ndf.test_n=test.Brand1

train.head()

test.head()

ndf

car_map = {car:i for i, car in enumerate(ndf.stack().unique())}
ndf['train_n'] = ndf['train_n'].map(car_map)
ndf['test_n'] = ndf['test_n'].map(car_map)

ndf

train['Brand1']=train['Brand1'].map(car_map)
test['Brand1']=test['Brand1'].map(car_map)

mdf=pd.DataFrame()

mdf['train_m']=" "

mdf['test_m']=" "

mdf.train_m=train.Model
mdf.test_m=test.Model

model_map = {model:i for i, model in enumerate(mdf.stack().unique())}
mdf['train_m'] = mdf['train_m'].map(model_map)
mdf['test_m'] = mdf['test_m'].map(model_map)

train['Model']=train['Model'].map(model_map)
test['Model']=test['Model'].map(model_map)

ldf=pd.DataFrame()

ldf['train_l']=" "

ldf['test_l']=" "

ldf.train_l=train.Location
ldf.test_l=test.Location
ldf

loc_map = {loc:i for i, loc in enumerate(ldf.stack().unique())}
ldf['train_l']= ldf['train_l'].map(loc_map)
ldf['test_l']= ldf['test_l'].map(loc_map)
ldf

train['Location']=train['Location'].map(loc_map)
test['Location']=test['Location'].map(loc_map)

train

test

fdf=pd.DataFrame()

fdf['train_f']=" "

fdf['test_f']=" "

fdf.train_f=train.Fuel_Type
fdf.test_f=test.Fuel_Type
fdf

fuel_map = {loc:i for i, loc in enumerate(fdf.stack().unique())}
fdf['train_f']= fdf['train_f'].map(fuel_map)
fdf['test_f']= fdf['test_f'].map(fuel_map)
fdf

train['Fuel_Type']=train['Fuel_Type'].map(fuel_map)
test['Fuel_Type']=test['Fuel_Type'].map(fuel_map)

test

train

tdf=pd.DataFrame()

tdf['train_t']=" "
tdf['test_t']=" "

tdf.train_t=train.Transmission
tdf.test_t=test.Transmission
tdf

trans_map = {trans:i for i, trans in enumerate(tdf.stack().unique())}
tdf['train_t']= tdf['train_t'].map(trans_map)
tdf['test_t']= tdf['test_t'].map(trans_map)
tdf

train['Transmission']=train['Transmission'].map(trans_map)
test['Transmission']=test['Transmission'].map(trans_map)

odf=pd.DataFrame()

odf['train_o']=" "
odf['test_o']=" "

odf.train_o=train.Owner_Type
odf.test_o=test.Owner_Type
odf

own_map = {own:i for i, own in enumerate(odf.stack().unique())}
odf['train_o']= odf['train_o'].map(own_map)
odf['test_o']= odf['test_o'].map(own_map)
odf

train['Owner_Type']=train['Owner_Type'].map(own_map)
test['Owner_Type']=test['Owner_Type'].map(own_map)

train

test

#creating Age of car column in train data
train['Age']=2020-train['Year']
train

#creating Age of car column in test data
test['Age']=2020-test['Year']
test

test_copy=test
test_copy

del test_copy['Version']

del test_copy['Brand']

del test_copy['Year']

from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

# Assuming df3_inputs contains your input features and df3_target contains your target variable

# Initialize the model
model = ExtraTreesRegressor()

# Fit the model
model.fit(df3_inputs, df3_target)

# Use inbuilt class feature_importances of ExtraTreeRegressor
feat_importances = pd.Series(model.feature_importances_, index=df3_inputs.columns)

# Plot graph of feature importances for better visualization
plt.figure(figsize=(11, 5))
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Features Importance")
feat_importances.nlargest(10).plot(kind='barh', color='#D98880')  # You can change the color as desired
plt.grid(color='black', linestyle='-.', linewidth=0.7)
plt.show()

test_copy

test_copy.head()

#Splitting data into independent and dependent data

#TAKE X AS TARINING DATA,
X=train[['Location','Kilometers_Driven','Fuel_Type','Transmission','Owner_Type','Mileage','Engine','Power','Seats','Brand1','Model','Age']]
y=train.Selling_Price

"""<h2> CROSS VALIDATION </H2>"""

#splitting the train-test data for validation
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=123)

"""<h1> TRAINING THE DATA USING VARIOUS REGRESSION MODELS </h1>"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import metrics
import xgboost as xgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error as mse
import numpy as np

# Perform train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=123)

# Function for Predicting accuracy
def Scr(Algo, X_data, y_data):
    scr = Algo.score(X_data, y_data)
    return scr

# K-fold cross-validation
def k_fold(model, X_data, y_data):
    k_folds = KFold(n_splits=5, shuffle=False)
    scores = cross_val_score(model, X_data, y_data, cv=k_folds, scoring='r2')
    avg_score = np.mean(scores)
    return avg_score

# Function for root mean square
def rmsle(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

# Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
y_pred_ls = linreg.predict(X_test)
print('Linear Regression:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred_ls), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred_ls), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_ls)), 2))
print('R²:', round(linreg.score(X_test, y_test), 4))

# Random Forest Regression
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred_reg = regressor.predict(X_test)
print('\nRandom Forest Regression:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred_reg), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred_reg), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_reg)), 2))
print('R²:', round(regressor.score(X_test, y_test), 4))

# Ridge Regression
rr = Ridge()
rr.fit(X_train, y_train)
y_pred_rr = rr.predict(X_test)
print('\nRidge Regression:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred_rr), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred_rr), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_rr)), 2))
print('R²:', round(rr.score(X_test, y_test), 4))

# GBR
gbr = GradientBoostingRegressor(loss='squared_error', max_depth=6)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
print('\nGradient Boosting Regression:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred_gbr), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred_gbr), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred_gbr)), 2))
print('R²:', round(gbr.score(X_test, y_test), 4))

# Lasso
ls = Lasso()
ls.fit(X_train, y_train)
ls_predic = ls.predict(X_test)
print('\nLasso:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, ls_predic), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, ls_predic), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, ls_predic)), 2))
print('R²:', round(ls.score(X_test, y_test), 4))

# XGB
model_xgb = xgb.XGBRegressor(colsample_bytree=0.52, gamma=0.03,
                             learning_rate=0.072, max_depth=6,
                             min_child_weight=2, n_estimators=2200,
                             reg_alpha=0, reg_lambda=1,
                             subsample=0.615,
                             random_state=7, nthread=-1)
model_xgb.fit(X_train, y_train)
xg_pre = (model_xgb.predict(X_test))
print('\nXGBoost:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, xg_pre), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, xg_pre), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, xg_pre)), 2))
print('R²:', round(model_xgb.score(X_test, y_test), 4))

# KNN
knn = KNeighborsRegressor(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('\nK-Nearest Neighbors:')
print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
accuracy = knn.score(X_test, y_test)
print('R²:', round(accuracy, 4))

test_copy.dtypes

X.dtypes

"""<h1>FINAL PREDICTION<h1> <h2>USIING XGB MODLE</h2>"""

prediction =model_xgb.predict(test_copy)
sol = pd.DataFrame({'Selling_Price': prediction })
sol = round(sol['Selling_Price'],2)

output = pd.read_excel('test1.xlsx')

output['Predicted Selling_price in lacs']=prediction
output['Predicted Selling_price in lacs']=output['Predicted Selling_price in lacs'].round(2)

writer = pd.ExcelWriter('Test_Final.xlsx', engine='xlsxwriter')
output.to_excel(writer, sheet_name='Sheet1', index=False)
#writer.save()

output.head(50)






