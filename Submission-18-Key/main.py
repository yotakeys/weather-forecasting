# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    
    X_test_desc = pd.DataFrame()
    X_desc = pd.DataFrame()
    
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
        
        # self.X_test_desc = self.X_test.describe()
        self.X_desc = self.X.describe()
        
        self.output_desc = pd.DataFrame(self.predictions).describe()
        
        
    def preProcessing(self):
     
        # test data
        self.test_clean = self.test_data.fillna(self.test_data.median().iloc[0])
        # train data
        self.train_clean = self.train_data.dropna(subset = [self.y_str])
        # self.train_clean = self.train_clean.fillna(self.train_clean.median().iloc[0])
        self.train_clean = self.train_clean.dropna()
        
      
    def featureEngineering(self):
        
        self.train_clean = self.train_clean[['time', 'elevation','city', self.y_str]]
        
        self.train_hourly_data['date'] = self.train_hourly_data.apply(lambda row: row.time.split('T')[0], axis = 1)
        train_mean_hourly = self.train_hourly_data.groupby(['date','city']).mean().reset_index().rename(columns = {'date' : 'time'})

        self.X = pd.merge(self.train_clean, train_mean_hourly, how='left', on = ['time','city'], suffixes = ['_d','_h'] )
        
        self.X['year'] = self.X.apply(lambda row: float(row.time.split('-')[0]), axis=1)
        self.X['moon'] = self.X.apply(lambda row: float(row.time.split('-')[1]), axis=1)
        # self.X['date'] = self.X.apply(lambda row: float(row.time.split('-')[2]), axis=1)
        
        # self.X['hour_sunrise'] = self.X.apply(lambda row: float(row['sunrise (iso8601)'].split('T')[1].split(':')[0]), axis=1)
        # self.X['minute_sunrise'] = self.X.apply(lambda row: float(row['sunrise (iso8601)'].split('T')[1].split(':')[1]), axis=1)
        # self.X['hour_sunset'] = self.X.apply(lambda row: float(row['sunset (iso8601)'].split('T')[1].split(':')[0]), axis=1)
        # self.X['minute_sunset'] = self.X.apply(lambda row: float(row['sunset (iso8601)'].split('T')[1].split(':')[1]), axis=1)
        
        from sklearn.preprocessing import OneHotEncoder
        
        OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encode_city_train = pd.DataFrame(OH_encoder.fit_transform(self.X[['city']])).add_prefix('city_')
        encode_city_train.index = self.X.index
        # self.X = pd.concat([self.X, encode_city_train], axis=1)
        
        self.X = self.X.select_dtypes(exclude='object')
        if self.mode:
            self.X = self.X.select_dtypes(exclude='object').drop([self.y_str], axis = 1)
        self.Y = self.train_clean[self.y_str]
            
        if self.mode:
            self.test_clean = self.test_clean[['id','time', 'elevation','city']]
            
            self.test_hourly_data['date'] = self.test_hourly_data.apply(lambda row: row.time.split('T')[0], axis = 1)
            test_mean_hourly = self.test_hourly_data.groupby(['date','city']).mean().reset_index().rename(columns = {'date' : 'time'})
    
            self.X_test = pd.merge(self.test_clean, test_mean_hourly, how='left', on = ['time','city'], suffixes = ['_d','_h'] )
            
            self.X_test['year'] = self.X_test.apply(lambda row: float(row.time.split('-')[0]), axis=1)
            self.X_test['moon'] = self.X_test.apply(lambda row: float(row.time.split('-')[1]), axis=1)
            # self.X_test['date'] = self.X_test.apply(lambda row: float(row.time.split('-')[2]), axis=1)
            
            # self.X_test['hour_sunrise'] = self.X_test.apply(lambda row: float(row['sunrise (iso8601)'].split('T')[1].split(':')[0]), axis=1)
            # self.X_test['minute_sunrise'] = self.X_test.apply(lambda row: float(row['sunrise (iso8601)'].split('T')[1].split(':')[1]), axis=1)
            # self.X_test['hour_sunset'] = self.X_test.apply(lambda row: float(row['sunset (iso8601)'].split('T')[1].split(':')[0]), axis=1)
            # self.X_test['minute_sunset'] = self.X_test.apply(lambda row: float(row['sunset (iso8601)'].split('T')[1].split(':')[1]), axis=1)
            
            encode_city_test = pd.DataFrame(OH_encoder.transform(self.X_test[['city']]))
            encode_city_test.index = self.X_test.index
            # self.X_test = pd.concat([self.X_test, encode_city_test], axis=1)
            
            self.X_test = self.X_test.select_dtypes(exclude='object').drop(['id'], axis = 1)
    
        
        
    def xgBoost(self):
        
        from sklearn.ensemble import RandomForestRegressor 
        self.model = RandomForestRegressor()
        
        # from xgboost import XGBRegressor
        
        # self.model = XGBRegressor()
        
        def mines(x):
            if x < 0:
                return 0
            return x
        
        if self.mode:
            self.model.fit(self.X, self.Y)
            self.predictions = pd.Series(self.model.predict(self.X_test)).apply(mines)

            self.output = pd.DataFrame({'id': self.test_data.id,
                        'rain_sum (mm)': self.predictions})
            self.output.to_csv('submission-18-key.csv', index=False)
            
            
        else:
            # from sklearn.model_selection import train_test_split
            # x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size= 0.8, random_state=2)
            
            train = self.X[self.X['year']<2021]
            test = self.X[self.X['year'] >=2021]
            
            x_train = train.drop([self.y_str], axis=1)
            y_train = train[self.y_str]
            
            x_test = test.drop([self.y_str], axis=1)
            y_test = test[self.y_str]
            
            self.model.fit(x_train, y_train)
            
            self.predictions = self.model.predict(x_test)
            self.predictions = pd.Series(self.model.predict(x_test)).apply(mines)
        
            from sklearn.metrics import mean_squared_error
            print("MSE XGB: ", mean_squared_error(y_test,self.predictions))
    
    
if __name__ == '__main__':
    
    Mod = WeatherForecast(mode = 1)