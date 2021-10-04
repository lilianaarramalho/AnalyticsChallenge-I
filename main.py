import pandas as pd
import pandas_profiling
import numpy
from sklearn.impute import KNNImputer
import scipy
from sklearn.preprocessing import MinMaxScaler
from function import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn import metrics

#MAIN CODE
train_set, test_set = import_data()
train_set,data,target = data_cleaning(train_set)
logistic_regression(train_set,data,target)
lac(train_set,data,target)
data=data.iloc[:,:4]
gradient_booster(data,target)



