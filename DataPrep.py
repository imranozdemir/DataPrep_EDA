#Attach packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class MyHelpers:

    def outlier_thresholds(self, dataframe, col_name, q1=0.05, q3=0.95):
        #Computing outliers lower and upper bound using the Inter Quantile Range(IQR)
        #You can change quantiles q1 & q3
        quartile1 = dataframe[col_name].quantile(q1)
        quantile3 = dataframe[col_name].quantile(q3)
        interquantile_range =quantile3 - quartile1
        up_limit = quantile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 *interquantile_range
        return low_limit, up_limit

    def check_outlier(self, dataframe, col_name, q1=0.05, q3=0.95):
        #Using outlier_thresholds function to check outliers
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name]< low_limit)].any(axis=None):
            return True
        else:
            return False

    def grap_outliers(self, dataframe, col_name, index=False):
        #Detecting outliers using outlier_thresholds function
        #Change index=True to access index value of outliers
        low, up = outlier_thresholds(dataframe, col_name)
        if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up ))].shape[0] >10:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up ))].head())
        else:
            print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

        if index:
            outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
            return outlier_index


    def replace_with_thresholds(self, dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe)]