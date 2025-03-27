import os
import pandas as pd


def drop_metadata(df):
    return df.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API', 'vt_detection', "VT_Malware_Deteccao", "AZ_Malware_Deteccao", 'CLASS'])


def save_results(file_name:str, title:str, content:list):
    os.makedirs('./results', exist_ok=True)

    f = open(f"./results/{file_name}.txt", "a")
    f.write(f'{title}\n')
    for c in content:
        f.write(f'{c}\n')
    print(f"Escrita dos resultados de {title} realizada\n")
    f.close()


vrs100k = pd.read_csv("variances100k.csv", sep=";", decimal=",")
def drop_low_var_100k(df, var=0.1):
    return df.drop(vrs100k.loc[vrs100k["variancia"] <= var, "column"], axis=1)


def drop_low_var_by_col_100k(df, col=4000) -> pd.DataFrame:
    return df.drop(vrs100k.nsmallest(len(vrs100k) - col, 'variancia')["column"], axis=1)


vrs1m = pd.read_csv("variances1m.csv", sep=";", decimal=",")
def get_high_var_by_col_1m(col=4000):
    return vrs1m.nlargest(col, 'variancia')['column']
