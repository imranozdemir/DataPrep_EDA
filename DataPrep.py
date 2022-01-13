#Attach packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class MyHelpers:

    def grap_col_names(self, dataframe, cat_th=10, car_th=20):
       """
       Grouping numeric columns, categoric columns, categoric but cardinal columns

       :param dataframe: dataframe
       :param cat_th: int, optinal
       :param car_th: int, optinal
       :return: cat_cols, num_cols and cat_but_car as a list
       """
       #cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                       and dataframe[col] != "0"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "0"]
        cat_cols= cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        #num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "0"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cols: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")
        return cat_cols, num_cols, cat_but_car


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
        #Replacing outliers with thresholds
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def remove_outlier(self, dataframe, col_name):
        #Removing outliers
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
        return df_without_outliers

    def missing_values_table(self, dataframe, na_name = False):
        #Number and ratio of missing values
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 1)], axis=1, keys=["n_miss", "ratio"])
        print(missing_df, end="\n")

        if na_name:
           return na_columns

    def missing_vs_target(self, dataframe, na_columns):
        #Allowing us to see the importance of missing values to the target features
        #Important function for feature engineering
        temp_df = dataframe.copy()
        for col in na_columns:
            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
            #adding binary flags for missing values
        na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
        #Pick flag added features
        for col in na_flags:
            #Do not forget to define target
            print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                                "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


    def label_encoder(self, dataframe, binary_col):
        # Label Encoding Function
        # define binary columns in binary_col
        labelencoder = LabelEncoder()
        dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
        return dataframe

    def one_hot_encoder(self, dataframe, categorical_cols, drop_first=False):
        #If you want to apply one-hot-encoding to all categorical variables, drop_first should be True
        dataframe = pd.get_dummies(dataframe, columns= categorical_cols, drop_first=drop_first)
        return dataframe