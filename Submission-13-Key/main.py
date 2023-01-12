# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# train_data = pd.read_csv('../weather-forecasting-datavidia/train.csv', sep=',')

# train_data['time_sunrise'] = train_data.apply(lambda row: row['sunrise (iso8601)'].split('T')[1], axis = 1)
# train_data['time_sunset'] = train_data.apply(lambda row: row['sunset (iso8601)'].split('T')[1], axis = 1)

# def convertTime(time):
#     times = [float(x) for x in time.split(':')]
#     return times[0] + (times[1] / 60)

# train_data['sunrise_convert'] = train_data['time_sunrise'].apply(convertTime)
# train_data['sunset_convert'] = train_data['time_sunset'].apply(convertTime)

# def convertDate(date):
#     dates = [float(x) for x in date.split('-')]
#     return (10 * (dates[0]%2000)) + dates[1] + (0.1 * dates[2])

# train_data['date_convert'] = train_data['time'].apply(convertDate)

# train_desc = train_data.describe()
# train_desc_obj = train_data.describe(include='object')

# sunrise_mean = train_data[['sunrise_convert', 'rain_sum (mm)']].groupby('sunrise_convert').mean().sort_values(by=['rain_sum (mm)'])
# sunset_mean = train_data[['sunset_convert', 'rain_sum (mm)']].groupby('sunset_convert').mean().sort_values(by=['rain_sum (mm)'])
# date_mean = train_data[['date_convert', 'rain_sum (mm)']].groupby('date_convert').mean().sort_values(by=['rain_sum (mm)'])

# plt.plot(sunrise_mean)
# plt.plot(sunset_mean)
# plt.plot(date_mean)
# plt.scatter(train_data['time_sunset'], train_data['rain_sum (mm)'])
# plt.show()

class WeatherForecast:

    # data var
    y_str = 'rain_sum (mm)'
    train_data = pd.DataFrame()
    train_hourly_data = pd.DataFrame()
    test_data = pd.DataFrame()
    test_hourly_data = pd.DataFrame()

    train_clean = pd.DataFrame()
    test_clean = pd.DataFrame()
    
    X_test = pd.DataFrame()
    
    X = pd.DataFrame()
    Y = pd.DataFrame()
    
    model = None
    
    predictions = np.ndarray(1)
    output = pd.DataFrame()
    
    mode = 0 
    
    # exploratory var
    train_data_desc = pd.DataFrame()
    test_data_desc = pd.DataFrame()
    train_hourly_data_desc = pd.DataFrame()
    test_hourly_data_desc = pd.DataFrame()
    train_data_obj_desc = pd.DataFrame()
    test_data_obj_desc = pd.DataFrame()
    train_hourly_data_obj_desc = pd.DataFrame()
    test_hourly_data_obj_desc = pd.DataFrame()
    
    train_clean_desc = pd.DataFrame()
    test_clean_desc = pd.DataFrame()
    
    output_desc = pd.DataFrame()
    
    def __init__(self, mode=0):
        self.mode = mode
        self.importData()
        self.preProcessing()
        self.featureEngineering()
        
        self.xgBoost()
        
        self.exploratoryData()
    
    
    def importData(self):
        self.train_data = pd.read_csv('../weather-forecasting-datavidia/train.csv', sep=',')
        self.train_hourly_data = pd.read_csv('../weather-forecasting-datavidia/train_hourly.csv', sep=',')
    
        self.test_data = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')
        self.test_hourly_data = pd.read_csv('../weather-forecasting-datavidia/test_hourly.csv', sep=',')


    def exploratoryData(self):
        self.train_data_desc = self.train_data.describe()
        self.test_data_desc = self.test_data.describe()
        self.train_hourly_data_desc = self.train_hourly_data.describe()
        self.test_hourly_data_desc = self.test_hourly_data.describe()
        self.train_data_obj_desc = self.train_data.describe(include=['object'])
        self.test_data_obj_desc = self.test_data.describe(include=['object'])
        self.train_hourly_data_obj_desc = self.train_hourly_data.describe(include=['object'])
        self.test_hourly_data_obj_desc = self.test_hourly_data.describe(include=['object'])
    
        self.train_clean_desc = self.train_clean.describe()
        self.test_clean_desc = self.test_clean.describe()
        
        self.output_desc = pd.DataFrame(self.predictions).describe()
        
        
    def preProcessing(self):
     
        # test data
        self.test_clean = self.test_data.fillna(self.test_data.median().iloc[0])
        
        # train data
        self.train_clean = self.train_data.dropna(subset = [self.y_str])
        self.train_clean = self.train_clean.fillna(self.train_clean.median().iloc[0])
        
      
    def featureEngineering(self):

        self.train_hourly_data['date'] = self.train_hourly_data.apply(lambda row: row.time.split('T')[0], axis = 1)
        train_mean_hourly = self.train_hourly_data.groupby(['date','city']).mean().reset_index().rename(columns = {'date' : 'time'})

        self.X = pd.merge(self.train_clean, train_mean_hourly, how='left', on = ['time','city'], suffixes = ['_d','_h'] )
        
        testing_city = self.X[['city',self.y_str]].groupby(['city']).mean().sort_values(by=[self.y_str]).reset_index()
        testing_city['id'] = pd.DataFrame([1,2,3,4,5,6,7,8,9,10])
        self.X['city_sort_id'] = self.X['city'].map(testing_city.set_index(['city'])['id'])
        self.X['city_rain_mean'] = self.X['city'].map(testing_city.set_index(['city'])[self.y_str])
        
        self.X['time_sunrise'] = self.X.apply(lambda row: row['sunrise (iso8601)'].split('T')[1], axis = 1)
        self.X['time_sunset'] = self.X.apply(lambda row: row['sunset (iso8601)'].split('T')[1], axis = 1)

        def convertTime(time):
            times = [float(x) for x in time.split(':')]
            return times[0] + (times[1] / 60)

        self.X['sunrise_convert'] = self.X['time_sunrise'].apply(convertTime)
        self.X['sunset_convert'] = self.X['time_sunset'].apply(convertTime)

        def convertDate(date):
            dates = [float(x) for x in date.split('-')]
            return (10 * (dates[0]%2000)) + dates[1] + (0.1 * dates[2])

        self.X['date_convert'] = self.X['time'].apply(convertDate)

        self.X = self.X.select_dtypes(exclude='object').drop([self.y_str], axis = 1)
        self.Y = self.train_clean[self.y_str]
        
        if self.mode:
            
            self.test_hourly_data['date'] = self.test_hourly_data.apply(lambda row: row.time.split('T')[0], axis = 1)
            test_mean_hourly = self.test_hourly_data.groupby(['date','city']).mean().reset_index().rename(columns = {'date' : 'time'})

            self.X_test = pd.merge(self.test_clean, test_mean_hourly, how='left', on = ['time','city'], suffixes = ['_d','_h'] )
            
            self.X_test['city_sort_id'] = self.X_test['city'].map(testing_city.set_index(['city'])['id'])
            self.X_test['city_rain_mean'] = self.X_test['city'].map(testing_city.set_index(['city'])[self.y_str])
            
            self.X_test['time_sunrise'] = self.X_test.apply(lambda row: row['sunrise (iso8601)'].split('T')[1], axis = 1)
            self.X_test['time_sunset'] = self.X_test.apply(lambda row: row['sunset (iso8601)'].split('T')[1], axis = 1)

            self.X_test['sunrise_convert'] = self.X_test['time_sunrise'].apply(convertTime)
            self.X_test['sunset_convert'] = self.X_test['time_sunset'].apply(convertTime)

            self.X_test['date_convert'] = self.X_test['time'].apply(convertDate)
            
            self.X_test = self.X_test.select_dtypes(exclude='object').drop(['id'], axis = 1)
        
        
        
    def xgBoost(self):
        
        from xgboost import XGBRegressor
        
        self.model = XGBRegressor()
        
        def mines(x):
            if x < 0:
                return 0
            return x
        
        if self.mode:
            self.model.fit(self.X, self.Y)
            self.predictions = pd.Series(self.model.predict(self.X_test)).apply(mines)

            self.output = pd.DataFrame({'id': self.test_data.id,
                        'rain_sum (mm)': self.predictions})
            self.output.to_csv('submission-13-key.csv', index=False)
            
            
        else:
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size= 0.75, random_state=200)
            self.model.fit(x_train, y_train)
            
            self.predictions = self.model.predict(x_test)
            self.predictions = pd.Series(self.model.predict(x_test)).apply(mines)
        
            from sklearn.metrics import mean_squared_error
            print("MSE XGB: ", mean_squared_error(y_test,self.predictions))
    
    
if __name__ == '__main__':
    
    Mod = WeatherForecast(mode = 1)