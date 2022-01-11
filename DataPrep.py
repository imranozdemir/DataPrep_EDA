#Attach packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class MyHelpers:

    def outlier_thresholds(self, dataframe, col_name, q1=0.05, q3=0.95):
        #you can change quantiles q1&q3
        quartile1 = dataframe[col_name].quantile(q1)
        quantile3 = dataframe[col_name].quantile(q3)
        interquantile_range =quantile3 - quartile1
        up_limit = quantile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 *interquantile_range
        return low_limit, up_limit

