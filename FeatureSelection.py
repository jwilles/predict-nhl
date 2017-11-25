from Model import load_features_d, load_features_o, load_target_d, load_target_o
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def corr_o():
    X = load_features_o()
    y = load_target_o()
    cols = list(X)
    for i in range(len(cols)):
        print cols[i], pearsonr(np.array(X.iloc[:,i]).ravel(),np.array(y).ravel())[0]


def corr_d():
    X = load_features_d()
    y = load_target_d()
    cols = list(X)
    for i in range(len(cols)):
        print cols[i], pearsonr(np.array(X.iloc[:,i]).ravel(),np.array(y).ravel())[0]

def salary_plot():
    X = load_features()['salary']
    y = load_target()
    plt.figure()
    plt.scatter(X,y,s=3, color='red')
    plt.xlabel('Salary [$]')
    plt.ylabel('Ownership Percentage [%]')
    plt.title('Salary')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/salary.eps', format='eps', dpi=1000)

def twitter_plot():
    X = load_features()['tweetcount']
    y = load_target()
    plt.figure()
    plt.scatter(X,y,s=3, color='blue')
    plt.xlabel('Tweet Count')
    plt.ylabel('Ownership Percentage [%]')
    plt.title('Tweet Count')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/tweet.eps', format='eps', dpi=1000)

def propoints_plot():
    X = load_features()['proj_points']
    y = load_target()
    plt.figure()
    plt.scatter(X,y,s=3, color='green')
    plt.xlabel('Projected Points')
    plt.ylabel('Ownership Percentage [%]')
    plt.title('Projected Points')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/points.eps', format='eps', dpi=1000)

def home_plot():
    X = load_features()['home']
    y = load_target()
    plt.figure()
    plt.scatter(X,y,s=3, color='black')
    plt.xlabel('Home or Away')
    plt.ylabel('Ownership Percentage [%]')
    plt.title('Home or Away')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/home.eps', format='eps', dpi=1000)

def dpoints_plot():
    X = load_features()['d_points']
    y = load_target()
    plt.figure()
    plt.scatter(X,y,s=3, color='black')
    plt.xlabel('Number of Projected Points for Opposing Defense')
    plt.ylabel('Ownership Percentage [%]')
    plt.title('Number of Projected Points for Opposing Defense')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/d_points.eps', format='eps', dpi=1000)

def tsne_plot():
    X_embedded = TSNE(n_components=2).fit_transform(load_features())
    X_embedded = X_embedded.shape
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X_1 = np.array(X_embedded[:,0].tolist()).ravel()
    X_2 = np.array(X_embedded[:,1].tolist()).ravel()
    Y = np.array(load_target()).ravel()
    ax.scatter(X_1, X_2, Y, c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Ownership Percentage')
    plt.show()
    plt.savefig('/Users/user/COMS-6998-REPO/FinalProject/FeaturePlots/tsne.eps', format='eps', dpi=1000)


