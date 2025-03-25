import os
import math
import numpy
import pandas as pd

from ML_algs.utils import drop_metadata
from ML_algs.utils import drop_low_var_by_col


def build_fragments(overwrite=False):
    if not os.path.isdir('./1m_fragments_for_100k_model'):
        os.mkdir('./1m_fragments_for_100k_model')

    if len(os.listdir("./1m_fragments_for_100k_model")) > 0 and not overwrite:
        return

    df = pd.read_hdf('./dataset.h5')
    sha256s = df['SHA256'].str.upper()
    df = drop_low_var_by_col(drop_metadata(df))

    data = numpy.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)

    dataset = data['data']

    rows, _ = dataset.shape
    new_df = pd.DataFrame({'SHA256': data['sha256'], 'CLASS': data['classes'], 'vt_detection': data['metadata'][:,6]})
    new_df['SHA256'] = new_df['SHA256'].astype('U')
    new_df['CLASS'] = new_df['CLASS'].astype('B')
    new_df['vt_detection'] = new_df['vt_detection'].astype('B')

    def get_col(column):
        category, attribute = column.lower().split("::")
        if category == 'apicall':
            attribute = attribute[:-2]
        new_column = f"{category}s::{attribute}"
        idx = numpy.nonzero(data['column_names'] == new_column)[0]
        return pd.DataFrame({column: dataset[:, idx[0]] if len(idx) > 0 else [0 for _ in range(rows)]})

    new_df = pd.concat([new_df, pd.concat([get_col(column) for column in df.columns], axis=1)], axis=1)

    new_df = new_df[~new_df["SHA256"].isin(sha256s)]

    for i in range(math.ceil(rows / 100_000)):
        new_df[i * 100_000:(i * 100_000) + 100_000].to_hdf(
            f"./1m_fragments_for_100k_model/fragment_{i + 1}.h5",
            key="df",
            mode="w",
            format="f"
        )


def run(overwrite_fragments=False):
    build_fragments(overwrite=overwrite_fragments)


def main():
    run(
        overwrite_fragments=True,
        dataset_1m=None
    )


if __name__ == "__main__":
    main()
