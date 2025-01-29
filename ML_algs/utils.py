import os
import pandas as pd


def save_results(file_name:str, title:str, content:list):
    os.makedirs('./results', exist_ok=True)

    f = open(f"./results/{file_name}.txt", "a")
    f.write(f'{title}\n')
    for c in content:
        f.write(f'{c}\n')
    print(f"Escrita dos resultados de {title} realizada\n")
    f.close()


vrs = pd.read_csv("variances.csv", sep=";", decimal=",")
def drop_low_var(df, var=0.1):
    return df.drop(vrs.loc[vrs["variancia"] <= var, "column"], axis=1)
