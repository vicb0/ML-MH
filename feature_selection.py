import pandas as pd
import numpy as np

def feature_selection(file_name, variance):
    df = pd.read_csv(file_name, sep=';') 

    df["Variance"]=df["Variance"].str.replace(',','.')
    df["Variance"] = pd.to_numeric(df["Variance"])

    df = df.loc[df['Variance'] < variance]

    return df["Feature"].tolist()


#print(len(feature_selection('variances1.csv', 0.1)))
