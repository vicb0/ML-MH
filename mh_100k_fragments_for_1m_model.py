import os
import math
import numpy
import pandas as pd

from ML_algs.utils import drop_metadata
from ML_algs.utils import drop_low_var_by_col_100k
from ML_algs.utils import get_high_var_by_col_1m


def build_fragments(overwrite=False):
    if not os.path.isdir('./100k_fragments_for_1m_model'):
        os.mkdir('./100k_fragments_for_1m_model')

    if len(os.listdir("./100k_fragments_for_1m_model")) > 0 and not overwrite:
        return

    data = numpy.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    sha256s = map(lambda x: x.lower(), data['sha256'])
    
    headers = get_high_var_by_col_1m().sort_index()
    headers = headers + headers.groupby(headers).cumcount().astype(str).replace({'0':''})

    dataset = pd.read_hdf('./dataset.h5')
    rows, _ = dataset.shape

    new_df = pd.DataFrame({'SHA256': dataset['SHA256'], 'CLASS': dataset['CLASS'], 'vt_detection': dataset['vt_detection']})
    dataset = drop_metadata(dataset)

    dataset.columns = dataset.columns.str.lower()
    s = dataset.columns.to_series()
    dataset.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

    def get_col(column):
        category, attribute = column.lower().split("::")
        if category == 'apicalls':
            attribute = f'{attribute}()'
        new_column = f"{category[:-1]}::{attribute}"

        if new_column in dataset.columns:
            return dataset[new_column].to_list()
        return [0 for _ in range(rows)]

    new_df = pd.DataFrame({
        **new_df,
        **{column: get_col(column) for column in headers}
    })

    new_df = new_df[~new_df["SHA256"].isin(sha256s)]

    for i in range(math.ceil(rows / 10_000)):
        new_df[i * 10_000:(i * 10_000) + 10_000].to_hdf(
            f"./100k_fragments_for_1m_model/fragment_{i + 1}.h5",
            key="df",
            mode="w",
            format="f"
        )


def run(overwrite_fragments=False):
    build_fragments(overwrite=overwrite_fragments)


def main():
    run(
        overwrite_fragments=True
    )


if __name__ == "__main__":
    main()
