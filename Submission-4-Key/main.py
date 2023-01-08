# Import Modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class WeatherForecast:
    
    y_str = 'rain_sum (mm)'
    train_data = pd.DataFrame()
    train_hourly_data = pd.DataFrame()
    test_data = pd.DataFrame()
    test_hourly_data = pd.DataFrame()

    train_clean = pd.DataFrame()
    test_clean = pd.DataFrame()
    
    X_test = pd.DataFrame()
    
    X = pd.DataFrame()
    y = pd.DataFrame()
    
    predictions = np.ndarray(1)
    output = pd.DataFrame()
    
    def __init__(self):
        self.importData()
    
    def importData(self):
        self.train_data = pd.read_csv('../weather-forecasting-datavidia/train.csv', sep=',')
        self.train_hourly_data = pd.read_csv('../weather-forecasting-datavidia/train_hourly.csv', sep=',')
    
        self.test_data = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')
        self.test_hourly_data = pd.read_csv('../weather-forecasting-datavidia/test.csv', sep=',')

    def preProcessing_1(self):
        # train data
        self.train_clean = self.train_data.dropna(subset = [self.y_str])
        self.train_clean = self.train_clean.dropna(axis=1)
    
    def featureEngineering_1(self):
        # test data
        self.X_test = self.test_data.select_dtypes(exclude='object').drop(['id', 'winddirection_10m_dominant (°)'], axis = 1)
        
        self.X = self.train_clean.select_dtypes(exclude='object').drop('rain_sum (mm)', axis = 1)
        self.Y = self.train_clean[self.y_str]
    
    def randomForestTrain(self):
        
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size= 0.8)
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        
        rfregressor = RandomForestRegressor()
        
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]
        
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf}
        
        model = RandomizedSearchCV(estimator = rfregressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
        
        model.fit(x_train, y_train)
        
        self.predictions = model.predict(x_test)
        from sklearn.metrics import mean_squared_error
        print("MSE : ", mean_squared_error(y_test,self.predictions))
        
    def xgboostTrain(self):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.Y, train_size= 0.8)
        
        from xgboost import XGBRegressor
        model = XGBRegressor()
        
        model.fit(x_train, y_train)
        
        self.predictions = model.predict(x_test)
        from sklearn.metrics import mean_squared_error
        print("MSE : ", mean_squared_error(y_test,self.predictions))
        
    def randomForestTest(self):
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import RandomizedSearchCV
        
        rfregressor = RandomForestRegressor()
        
        n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 12)]
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
        min_samples_split = [2, 5, 10, 15, 100]
        min_samples_leaf = [1, 2, 5, 10]
        
        random_grid = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf}
        
        model = RandomizedSearchCV(estimator = rfregressor, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=1, n_jobs = 2)
        
        model.fit(self.X, self.Y)
        
        self.predictions = model.predict(self.X_test)
        self.output = pd.DataFrame({'id': self.test_data.id,
                        'rain_sum (mm)': self.predictions})
        self.output.to_csv('submission-4-key.csv', index=False)
    
    
if __name__ == '__main__':
    Mod = WeatherForecast()
    Mod.preProcessing_1()
    Mod.featureEngineering_1()
    Mod.randomForestTest()
    # Mod.xgboostTrain()
    
    now = pd.DataFrame(Mod.predictions)
    desc = now.describe()
        