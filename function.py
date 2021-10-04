import pandas as pd
import pandas_profiling
import numpy
from sklearn.impute import KNNImputer
import scipy
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV






def import_data():
    # READING CSV
    train_set = pd.read_csv('train.csv', encoding='ISO-8859-1', sep=",")
    test_set = pd.read_csv('test.csv', encoding='ISO-8859-1', sep=",")
    return train_set, test_set

def data_cleaning(train_set):

    #CHECK HOW MANY EMPTY VALUES EXIST
    print(train_set.isna().sum())

    #DROP NA
    #Cabin column has a 77 % of missing values and does not matter, we should drop the column
    #Embarked column has only 2 values missing, we should drop the rows
    train_set = train_set[train_set['Embarked'].notna()]
    train_set = train_set.drop(columns = ['Cabin'], axis = 1)

    #ADD COLUMN OF NUMBER OF RELATIVES
    #We considered that the aggregated number of relatives is more importan than the individual values
    train_set['Relatives'] = train_set['SibSp'] + train_set['Parch']
    #DROP COLUMNS THAT DO NOT MATTER TO THE SOLUTION
    #The ticket and name columns do not matter to the solution, as well as the SibSp and Parch ones,
    #now that we created the Relatives Column
    train_set = train_set.drop(columns=['Ticket', 'Name', 'SibSp', 'Parch'], axis=1)

    #CREATE BINARY/DUMMY VARIABLES
    #We should create binary/dummy variables for the categoric varibles (Sex and Embarked)
    train_set['Male'] = train_set['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    y=pd.get_dummies(train_set.Embarked, prefix='Embarked')
    train_set = pd.concat([train_set, y], axis=1)
    train_set = train_set.drop(columns = ['Sex', 'Embarked'], axis = 1)

    print(train_set.isna().sum())

    #DROP COLUMNS THAT DO NOT MATTER FOR IMPUTATION
    #(we know imputation is required because there are >4% missing values from 'Age' column)

    train_set_normalized=train_set.drop(columns=['PassengerId'],axis=1)

    #NORMALIZATION
    scaler=MinMaxScaler()
    train_set_normalized = pd.DataFrame(scaler.fit_transform(train_set_normalized), columns = train_set_normalized.columns)

    #KNN IMPUTER TO FILL NA
    imputer = KNNImputer(n_neighbors=5)
    train_set_normalized = pd.DataFrame(imputer.fit_transform(train_set_normalized),columns = train_set_normalized.columns)

    #GETTING PASSENGER ID
    #train_set = pd.merge(train_set_normalized, train_set['PassengerId'],left_on=train_set_normalized.index,right_on=train_set.index)
    train_set = train_set_normalized.copy()

    #PANDA PROFILLING
    #pandas_profile=train_set_normalized.profile_report()
    #pandas_profile.to_file(output_file="profile.html")

    #DROP FARE (HIGH CORRELATION W/ PCLASS)
    train_set=train_set.drop(columns='Fare',axis=1)

    train_set.to_csv('data.csv')

    print(train_set.isna().sum())

    data = train_set.iloc[:, 1:]
    target = train_set.iloc[:, 0]

    return train_set,data,target


def logistic_regression(train_set,data,target):

    # no of features
    nof_list=13
    high_score = 0
    # Variable to store the optimum features
    nof = 0
    score_list = []

    model=LogisticRegression()
    rfe = RFE(estimator=model, n_features_to_select=4)
    X_rfe = rfe.fit_transform(X=data, y=target)
    model.fit(X=X_rfe, y=target)
    print(rfe.support_)
    print(rfe.ranking_)

    selected_features = pd.Series(rfe.support_, index=data.columns)

    print(selected_features)

    for n in range(1,nof_list):
        # we are going to see in the next class this "train_test_split()"...
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0)

        model = LogisticRegression()
        rfe = RFE(estimator=model, n_features_to_select=n)
        X_train_rfe = rfe.fit_transform(X_train, y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe, y_train)

        score = model.score(X_test_rfe, y_test)
        score_list.append(score)

        if (score > high_score):
            high_score = score
            nof = n

    print("Optimum number of features: %d" % nof)
    print("Score with %d features: %f" % (nof, high_score))

def plot_importance(coef,name):
    imp_coef = coef.sort_values()
    plt.figure(figsize=(8,10))
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using " + name + " Model")
    plt.savefig('temp.png')

def lac(train_set,data,target):

    reg = LassoCV()
    reg.fit(X=data, y=target)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" % reg.score(X=data, y=target))
    coef = pd.Series(reg.coef_, index=data.columns)
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")
    print(coef.sort_values())
    plot_importance(coef, 'Lasso')

def display(results):
   print(f'Best parameters are: {results.best_params_}')
   print("\n")
   mean_score = results.cv_results_['mean_test_score']
   std_score = results.cv_results_['std_test_score']
   params = results.cv_results_['params']
   for mean,std,params in zip(mean_score,std_score,params):
       print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

def gradient_booster(data,target):

    X_train_val, X_test, y_train_val, y_test = train_test_split(data,target,test_size = 0.2,random_state = 40,shuffle = True,stratify = target)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,test_size = 0.25,random_state = 40,shuffle = True,stratify = y_train_val)

    print('train:{}% | validation:{}% | test:{}%'.format(round(len(y_train) / len(target), 2),
                                                         round(len(y_val) / len(target), 2),
                                                         round(len(y_test) / len(target), 2)
                                                         ))


    gb = GradientBoostingClassifier(subsample=1, min_samples_leaf=1, min_samples_split=80, learning_rate=0.1,
                                    max_depth=3, n_estimators=450, random_state=5, max_features=2)

    gb.fit(X_train, y_train)

    y_pred_gb = gb.predict(X_test)

    gb_score = gb.score(X_test, y_test)

    parameters = {
        "min_samples_split": [70, 80],
        "n_estimators": [500, 450, 550],
        "max_features": [1,2,3,4]

    }

    cv = GridSearchCV(gb, parameters, cv=5)
    cv.fit(data, target.values.ravel())

    display(cv)

    print(gb_score)
    print(classification_report(y_test, y_pred_gb))


















