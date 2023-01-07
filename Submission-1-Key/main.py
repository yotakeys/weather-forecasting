
import pandas as pd

train = pd.read_csv('../weather-forecasting-datavidia/train.csv', sep=',')
test = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')

train = train.select_dtypes(exclude='object').fillna(0)
x_train = train.drop('rain_sum (mm)', axis = 1)
y_train = train['rain_sum (mm)']

x_test = test.select_dtypes(exclude='object').fillna(0).drop(['id'], axis = 1)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

rain_predict = model.predict(x_test)

output = pd.DataFrame(
    { 'id' : test.id,
         'rain_sum (mm)' : rain_predict
     })

output.to_csv('submission-1-key.csv', index = False)