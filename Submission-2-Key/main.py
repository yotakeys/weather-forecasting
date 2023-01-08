# weather forecasting

# import modules
import pandas as pd
import matplotlib.pyplot as plt

# read data
data_train = pd.read_csv('../weather-forecasting-datavidia/train.csv', sep=',')
data_test = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')


train_clean = data_train.dropna(subset =['rain_sum (mm)'])
train_column_nan_sum = train_clean.isna().sum()
test_column_nan_sum = data_test.isna().sum()

train_clean = train_clean.dropna(axis=1)
train_clean  = train_clean[train_clean['rain_sum (mm)'] != train_clean['rain_sum (mm)'].max()]


test_clean = data_test.select_dtypes(exclude='object').drop(['id', 'winddirection_10m_dominant (°)'], axis = 1)
train_desc = train_clean.describe()
train_desc_obj = train_clean.describe(include='O')

# plt.scatter(train_clean.iloc[:,12], train_clean['rain_sum (mm)'])
#plt.show()

# from sklearn.model_selection import train_test_split
X_train = train_clean.select_dtypes(exclude='object').drop('rain_sum (mm)', axis = 1)
y_train = train_clean['rain_sum (mm)']

from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(X_train, y_train)

# from sklearn.metrics import mean_squared_error
# print("Mean Squared Error: " + str(mean_squared_error(predictions, y_valid)))

predictions = my_model.predict(test_clean)

def mines(x):
    if x < 0:
        return 0
    return x


output = pd.DataFrame({'id': data_test.id,
                        'rain_sum (mm)': predictions})
output['rain_sum (mm)'] = output['rain_sum (mm)'].apply(mines)
# print(output['rain_sum (mm)'].min())
output.to_csv('submission-2-key.csv', index=False)