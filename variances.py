import os

import numpy as np
import pandas as pd

from parser100k import get_headers as get_headers_100k


def generate_variances_100k(overwrite=False):
    if os.path.isfile("./variances100k.csv") and not overwrite:
        return

    df = pd.read_hdf("dataset.h5")
    df = df.drop(get_headers_100k("others"), axis=1)
    df = df.var()
    df = pd.DataFrame({'column': df.index, 'variancia': df.values})
    df.to_csv("variances100k.csv", sep=";", float_format="%.16f", decimal=",", index=False)


def generate_variances_1m(overwrite=False):
    if os.path.isfile("./variances1m.csv") and not overwrite:
        return

    data = np.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    headers = data['column_names']
    df = data['data']

    variances = []
    for i in range(len(headers)):
        column_values = df[:, i]
        variances.append(column_values.var())

    df = pd.DataFrame({'column': headers, 'variancia': variances})
    df.to_csv("variances1m.csv", sep=";", float_format="%.16f", decimal=",", index=False)


def run(overwrite_variances_100k=False, overwrite_variances_1m=False):
    generate_variances_100k(overwrite=overwrite_variances_100k)
    generate_variances_1m(overwrite=overwrite_variances_1m)


def main():
    run(
        overwrite_variances_100k=True,
        overwrite_variances_1m=True
    )


if __name__ == "__main__":
    main()
