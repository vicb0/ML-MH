import pandas as pd

def feature_selection(file_name, variance):
    df = pd.read_csv(file_name, sep=';')

    df.iloc[:, 1] = df.iloc[:, 1].str.replace(',', '.')
    df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1])

    df = df.loc[df.iloc[:, 1] < variance]

    return df.iloc[:, 0].tolist()


print(feature_selection('variances1.csv', 0.1))
