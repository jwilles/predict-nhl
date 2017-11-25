import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os
#from tweetClass import twitterFeatures

class ownershipPredict(object):
    #Load pre-processed csv from fan duels
    def load_fanduel(self):
        return pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/temp/fanduel_current_11_tweets.csv')

    # def add_tweets(self):
    #     path = os.path.dirname(os.path.realpath(__file__))
    #     featureExtract = twitterFeatures()
    #     featureExtract.minTweetCountswithSentiments(path, 'fanduel_current_11', 13, 18, 1, 6, 5)

    #Make predictions for offence
    def ownershipPredict_o(self, input):
        input = np.array(input).reshape(1,-1)
        path = os.path.dirname(os.path.realpath(__file__))
        theFile = '/offence.pkl'
        model = joblib.load(path + theFile)
        return model.predict(input)

    #Make predictions for defence
    def ownershipPredict_d(self, input):
        input = np.array(input).reshape(1,-1)
        path = os.path.dirname(os.path.realpath(__file__))
        theFile = '/defence.pkl'
        model = joblib.load(path + theFile)
        return model.predict(input)

    def return_project(self):
        df = self.load_fanduel()
        df = df.dropna()
        defence = df[df['D'] ==1]
        offence = df[df['D'] == 0]
        d_output = list()
        o_output = list()
        for x in range(len(offence)):
            input = [float(offence.salary.iloc[x]), offence.proj_points.iloc[x],float(offence.home.iloc[x]),offence.d_points.iloc[x],
                     offence.K.iloc[x], offence.QB.iloc[x], offence.RB.iloc[x], offence.WR.iloc[x], offence.TE.iloc[x], offence.doubtful.iloc[x],
                     offence.out.iloc[x], offence.pup.iloc[x], offence.questionable.iloc[x], float(offence.minTweetCounts.iloc[x]), float(offence.sentimentCounts.iloc[x])]
            o_output.append(self.ownershipPredict_o(input))

        for x in range(len(defence)):
            input = [defence.salary.iloc[x], defence.proj_points.iloc[x],defence.home.iloc[x], defence.minTweetCounts.iloc[x], defence.sentimentCounts.iloc[x]]
            d_output.append(self.ownershipPredict_d(input))
        defence['Predicted Ownership'] = np.array(d_output).ravel()
        offence['Predicted Ownership'] = np.array(o_output).ravel()
        df = pd.concat([offence,defence],axis=0)
        df.loc[df['D'] == 1, 'Position'] = 'D'
        df.loc[df['QB'] == 1, 'Position'] = 'QB'
        df.loc[df['K'] == 1, 'Position'] = 'K'
        df.loc[df['RB'] == 1, 'Position'] = 'RB'
        df.loc[df['TE'] == 1, 'Position'] = 'TE'
        df.loc[df['WR'] == 1, 'Position'] = 'WR'
        df.loc[df['Predicted Ownership'] < 0, 'Predicted Ownership'] = 0
        df = df.sort_values('Predicted Ownership', ascending=False)
        df.to_csv(os.path.dirname(os.path.realpath(__file__)) + '/temp/our_projections.csv')
        return df
