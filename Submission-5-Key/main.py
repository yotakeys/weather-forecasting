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
        self.test_hourly_data = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')


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
        
        from sklearn.preprocessing import OrdinalEncoder
        
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value= np.nan)
        
        self.X = self.train_clean.copy()
        
        s = (self.X.dtypes == 'object')
        object_cols = list(s[s].index)
        self.X[object_cols] = ordinal_encoder.fit_transform(self.X[object_cols])
        
        
        self.X = self.X.select_dtypes(exclude='object').drop([self.y_str], axis = 1)
        self.Y = self.train_clean[self.y_str]
        
        if self.mode:
            
            self.X_test = self.test_clean.copy()
            
            s = (self.X_test.dtypes == 'object')
            object_cols = list(s[s].index)
            
            self.X_test[object_cols] = ordinal_encoder.transform(self.X_test[object_cols])
            self.X_test = self.X_test.select_dtypes(exclude='object').drop(['id'], axis = 1)
        
        
        
    def xgBoost(self):
        
        from xgboost import XGBRegressor
        from sklearn.model_selection import RandomizedSearchCV
        
        xgbr = XGBRegressor()
        params = { 'max_depth': [3, 5, 6, 10, 15, 20],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': np.arange(0.5, 1.0, 0.1),
            'colsample_bytree': np.arange(0.4, 1.0, 0.1),
            'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
            'n_estimators': [100, 500, 1000]}
        
        self.model = RandomizedSearchCV(estimator=xgbr,
                          param_distributions=params,
                          scoring='neg_mean_squared_error',
                          n_iter=25,
                          verbose=1)
        
        def mines(x):
            if x < 0:
                return 0
            return x
        
        if self.mode:
            self.model.fit(self.X, self.Y)
            self.predictions = self.model.predict(self.X_test)
            self.predictions = pd.Series(self.predictions).apply(mines)
            
            self.output = pd.DataFrame({'id': self.test_data.id,
                        'rain_sum (mm)': self.predictions})
            self.output.to_csv('submission-5-key.csv', index=False)
            
            
        else:
            from sklearn.model_selection import train_test_split
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size= 0.8, random_state=20)
            self.model.fit(x_train, y_train)
            
            self.predictions = self.model.predict(x_test)
            
            self.predictions = pd.Series(self.predictions).apply(mines)
            from sklearn.metrics import mean_squared_error
            print("MSE XGB: ", mean_squared_error(y_test,self.predictions))
    
    
if __name__ == '__main__':
    
    Mod = WeatherForecast(mode = 1)
        