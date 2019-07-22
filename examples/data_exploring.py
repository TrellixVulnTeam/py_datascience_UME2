#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

data_path = '/media/user/Transcend/AI/Datasets'

def start():
    #bring in the six packs
    df_train = pd.read_csv(os.path.join(data_path, 'train.csv'))
    print(df_train.columns)
    print(df_train.head())
    print(df_train['SalePrice'].describe())
    sns.distplot(df_train['SalePrice'])
    plt.show()
    print("Skewness: %f " % df_train['SalePrice'].skew())
    print("Kurtosis: %f " % df_train['SalePrice'].kurt())
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x = var, y = 'SalePrice', ylim=(0, 800000))
    plt.show()


