import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score'''

'''from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor'''

data = pd.read_csv('./data.csv')
features = data.iloc[:, 1:8]
features.drop(['CGPA'], axis=1, inplace=True)
result = data.iloc[:, 8:9]

'''X_train, X_test, y_train, y_test = train_test_split(features, result,
                                                    random_state = 23,
                                                    test_size=0.5)
'''
#print(features.head())
reg = LinearRegression()
#reg.fit(X_train, y_train)
reg.fit(features, result)
'''pred = reg.predict(X_test)
acc = reg.score(X_test, y_test)
mse = mean_squared_error(y_test, pred)'''

import pickle
pickle.dump(reg, open("model.pkl","wb"))