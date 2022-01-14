from utility.DataPrep_EDA import MyHelpers
import pandas as pd


dataframe = pd.read_csv("C:/Users/lenovo/Desktop/train_identity.csv")
dataframe.head()

my_hepler_obj = MyHelpers()

my_hepler_obj.check_df(dataframe)

for col in dataframe.columns:
    my_hepler_obj.cat_summary(dataframe, col)

for col in dataframe.columns:
    my_hepler_obj.num_summary(dataframe, col)