import numpy as np
import pandas as pd
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.linear_model import Ridge, Lasso, SGDRegressor, ElasticNet, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR, SVR, NuSVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import r2_score
#import matplotlib as plt
import os
import sqlite3 as lite

class loadData(object):
    #Load LUT, features and
    def load_LUT(self):
        thePathLut = os.path.dirname(os.path.realpath(__file__))
        return pd.read_csv(thePathLut + '/predictorLUT.csv', index_col=0)  # ALGO LUT

    def load_o_data(self):
        thePathLut = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(thePathLut + '/finalfantasy.csv')
        df = df.loc[df['D'] == 0]
        df1 = df.loc[df['year'] == 2017]
        df2 = df.loc[df['year'] == 2016]
        df2 = df2.loc[df2['week'] >= 11]
        df = pd.concat([df1, df2], axis=0)
        df = df.loc[df['out'] == 0]
        df = df.loc[df['pup'] == 0]
        df = df.dropna()
        con = lite.connect('players.db')
        df.to_sql("master_training_o", con, if_exists='replace')
        return df

    def load_d_data(self):
        thePathLut = os.path.dirname(os.path.realpath(__file__))
        df = pd.read_csv(thePathLut + '/finalfantasy.csv')
        df = df.loc[df['D'] == 1]
        df1 = df.loc[df['year'] == 2017]
        df2 = df.loc[df['year'] == 2016]
        df2 = df2.loc[df2['week'] >= 11]
        df = pd.concat([df1, df2], axis=0)
        df = df.dropna()
        con = lite.connect('players.db')
        df.to_sql("master_training_d", con, if_exists='replace')

    def load_features_o(self):
        con = lite.connect('players.db')
        df1 = pd.read_sql_query("select * from master_training_o", con)
        df1 = df1[['salary', 'proj_points','home','d_points','K','QB','RB','WR','TE','doubtful','out','pup','questionable', 'minTweetCounts', 'sentimentCounts']].copy()
        df1['salary'] = df1['salary'].str.replace(',', '')
        df1['salary'] = df1['salary'].str.replace('$', '')
        df1['salary'] = df1['salary'].astype(float)
        return df1

    def load_features_d(self):
        con = lite.connect('players.db')
        df1 = pd.read_sql_query("select * from master_training_d ", con)
        df1 = df1[['salary', 'proj_points','home', 'minTweetCounts', 'sentimentCounts']].copy() #tweetCount, tweetSentiment
        df1['salary'] = df1['salary'].str.replace(',', '')
        df1['salary'] = df1['salary'].str.replace('$', '')
        df1['salary'] = df1['salary'].astype(float)
        return df1

    def load_target_o(self):
        con = lite.connect('players.db')
        df1 = pd.read_sql_query("select * from master_training_o", con)
        return df1[['ownership']].copy()

    def load_target_d(self):
        con = lite.connect('players.db')
        df1 = pd.read_sql_query("select * from master_training_d ", con)
        return df1[['ownership']].copy()

class offenceModel(object):

    def algoArray(self, theAlgo):
        load = loadData()
        theLUT = load.load_LUT()
        theAlgoOut = theLUT.loc[theAlgo, 'functionCall']
        return theAlgoOut

    def gridSearch(self, theAlgo):
        load = loadData()
        theLUT = load.load_LUT()
        theAlgoOut = theLUT.loc[theAlgo, 'gridSearch']
        return theAlgoOut

    #Optimize model from LUT using cross validation
    def optModel(self):
        load = loadData()
        featureMatrix = load.load_features_o()
        fullIndex = load.load_target_o()

        theModels = ['OLS','RR','LR','EN','GBR','RFR','BR','ETR']
        theResults = pd.DataFrame(0, index=theModels, columns=['accuracy', 'confidence', 'runtime'])

        for theModel in theModels:
            startTime = time.time()
            model = eval(self.algoArray(theModel))
            print(theModel)

            # cross validation
            cvPerf = cross_val_score(model, featureMatrix, fullIndex, cv=10)
            theResults.loc[theModel, 'accuracy'] = round(cvPerf.mean(), 2)
            theResults.loc[theModel, 'confidence'] = round(cvPerf.std() * 2, 2)
            endTime = time.time()
            theResults.loc[theModel, 'runtime'] = round(endTime - startTime, 0)

        print(theResults)

        bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
        modelChoice = theResults['accuracy'].idxmax()
        return modelChoice

    #Optimize parameters
    def optGrid(self, modelChoice):
        load = loadData()
        startTime = time.time()
        featureMatrix = load.load_features_o()
        fullIndex = load.load_target_o()

        model = eval(self.algoArray(modelChoice))
        grid = eval(self.gridSearch(modelChoice))
        grid.fit(featureMatrix, fullIndex)

        bestScore = round(grid.best_score_, 4)
        parameters = grid.best_params_
        endTime = time.time()
        print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime, 0)))
        return parameters

    #Opt
    def optFunc(self, theAlgo, theParams):
        load = loadData()
        theLUT = load.load_LUT()
        theModel = theLUT.loc[theAlgo, 'optimizedCall']
        tempParam = list()
        for key, value in theParams.iteritems():
            tempParam.append(str(key) + "=" + str(value))
        theParams = ",".join(tempParam)
        theModel = theModel + theParams + ")"
        return theModel

    #train OWNERSHIP model with optimal model and dump in pickle file
    def ownershipTrain(self):
        load = loadData()
        featureMatrix = load.load_features_o()
        fullIndex = load.load_target_o()
        modelChoice = self.optModel()
        parameters = self.optGrid(modelChoice)

        startTime = time.time()
        model = eval(self.optFunc(modelChoice, parameters))  # train fully validated and optimized model
        model.fit(featureMatrix, fullIndex)
        # model.fit(train,trainIndex)
        joblib.dump(model, 'offence.pkl')  # save model
        endTime = time.time()
        print("Model Save Time: " + str(round(endTime - startTime, 0)))

class defenceModel(object):

    def algoArray(self, theAlgo):
        load = loadData()
        theLUT = load.load_LUT()
        theAlgoOut = theLUT.loc[theAlgo, 'functionCall']
        return theAlgoOut

    def gridSearch(self, theAlgo):
        load = loadData()
        theLUT = load.load_LUT()
        theAlgoOut = theLUT.loc[theAlgo, 'gridSearch']
        return theAlgoOut

    #Optimize model from LUT using cross validation
    def optModel(self):
        load = loadData()
        featureMatrix = load.load_features_d()
        fullIndex = load.load_target_d()

        theModels = ['RR','LR','EN','GBR','RFR','BR','ETR']
        theResults = pd.DataFrame(0, index=theModels, columns=['accuracy', 'confidence', 'runtime'])

        for theModel in theModels:
            startTime = time.time()
            model = eval(self.algoArray(theModel))
            print(theModel)

            # cross validation
            cvPerf = cross_val_score(model, featureMatrix, fullIndex, cv=10)
            theResults.loc[theModel, 'accuracy'] = round(cvPerf.mean(), 2)
            theResults.loc[theModel, 'confidence'] = round(cvPerf.std() * 2, 2)
            endTime = time.time()
            theResults.loc[theModel, 'runtime'] = round(endTime - startTime, 0)

        print(theResults)

        bestPerfStats = theResults.loc[theResults['accuracy'].idxmax()]
        modelChoice = theResults['accuracy'].idxmax()
        return modelChoice

    #Optimize parameters
    def optGrid(self, modelChoice):
        load = loadData()
        startTime = time.time()
        featureMatrix = load.load_features_d()
        fullIndex = load.load_target_d()

        model = eval(self.algoArray(modelChoice))
        grid = eval(self.gridSearch(modelChoice))
        grid.fit(featureMatrix, fullIndex)

        bestScore = round(grid.best_score_, 4)
        parameters = grid.best_params_
        endTime = time.time()
        print("Best Score: " + str(bestScore) + " and Grid Search Time: " + str(round(endTime - startTime, 0)))
        return parameters

    #Opt
    def optFunc(self, theAlgo, theParams):
        load = loadData()
        theLUT = load.load_LUT()
        theModel = theLUT.loc[theAlgo, 'optimizedCall']
        tempParam = list()
        for key, value in theParams.iteritems():
            tempParam.append(str(key) + "=" + str(value))
        theParams = ",".join(tempParam)
        theModel = theModel + theParams + ")"
        return theModel

    #train OWNERSHIP model with optimal model and dump in pickle file
    def ownershipTrain(self):
        load = loadData()
        featureMatrix = load.load_features_d()
        fullIndex = load.load_target_d()
        modelChoice = self.optModel()
        parameters = self.optGrid(modelChoice)

        startTime = time.time()
        model = eval(self.optFunc(modelChoice, parameters))  # train fully validated and optimized model
        model.fit(featureMatrix, fullIndex)
        # model.fit(train,trainIndex)
        joblib.dump(model, 'defence.pkl')  # save model
        endTime = time.time()
        print("Model Save Time: " + str(round(endTime - startTime, 0)))



