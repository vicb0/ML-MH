import os
import math
import random
import pandas as pd

import numpy as np


def parse_headers(overwrite=False):
    if not os.path.isdir("./headers1m"):
        os.mkdir("./headers1m")

    data = np.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    headers = data['column_names']

    hs = ["apicalls", "intents", "permissions", "opcodes"]

    for h in hs:
        if os.path.isfile(f"./headers1m/{h}.txt") and not overwrite:
            continue
        with open(f"./headers1m/{h}.txt", "w+") as f:
            for header in headers.tolist():
                if header.lower().startswith(h):
                    f.write(header + "\n")

    headers = data['metadata_columns']

    if os.path.isfile("./headers1m/others.txt") and not overwrite:
        return
    with open("./headers1m/others.txt", "w+") as f:
        for header in headers.tolist():
            f.write(header + "\n")


def get_headers(header_type):
    f = open(f"./headers1m/{header_type}.txt", "r")
    return {line[:-1] for line in f.readlines()}


def split_1m(overwrite=False):
    if not os.path.isdir("./fragments1m"):
        os.mkdir("./fragments1m")

    if os.path.isfile('./fragments1m/fragment_1.npz') and not overwrite:
        return

    data = np.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    dataset = data['data']
    rows, _ = dataset.shape

    for i in range(math.ceil(rows / 100_000)):
        np.savez_compressed(
            file=f"./fragments1m/fragment_{i + 1}.npz",
            arr=dataset[i * 100_000: (i + 1) * 100_000]
        )


def build_hdf(overwrite=False):
    if os.path.isfile('./fragments1m/balanced_fragment.h5') and not overwrite:
        return

    data = np.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    headers = data['column_names'].tolist() + ['CLASS']
    vt_detection = data['metadata'][:,6]

    benign = []
    malware = []

    indeces = []
    for fragment in os.scandir('./fragments1m'):
        if not fragment.name.startswith('fragment_'):
            continue

        fragment_index = int(fragment.name.split(".")[0].split("_")[1])
        indeces.append(fragment_index)

    random.shuffle(indeces)

    SIZE = 100_000
    remaining_malware = remaining_benign = SIZE
    for c, index in enumerate(indeces):
        data = np.load(f'./fragments1m/fragment_{index}.npz')
        data = data['arr']

        if remaining_malware > 0:
            malwares = data[vt_detection[(index - 1) * 100_000:index * 100_000] >= 4]
            filter_indices = np.random.choice(range(len(malwares)), min([remaining_malware, remaining_malware // (len(indeces) - c), len(malwares)]), replace=False)
            remaining_malware -= len(filter_indices)
            malwares = np.take(malwares, filter_indices, axis=0)
            malwares = malwares.astype(np.uint8)
            malware.extend(malwares)

        if remaining_benign > 0:
            benigns = data[vt_detection[(index - 1) * 100_000:index * 100_000] < 4]
            filter_indices = np.random.choice(range(len(benigns)), min([remaining_benign, remaining_benign // (len(indeces) - c), len(benigns)]), replace=False)
            remaining_benign -= len(filter_indices)
            benigns = np.take(benigns, filter_indices, axis=0)
            benigns = benigns.astype(np.uint8)
            benign.extend(benigns)

        if len(benign) >= SIZE and len(malware) >= SIZE:
            break

    class_ = np.array([0]*len(benign) + [1]*len(malware), dtype=np.uint8).reshape(-1, 1)
    data = np.vstack([benign, malware])
    data = np.hstack([data, class_])

    del class_
    del malware
    del benign
    del vt_detection

    headers = pd.Series(headers)
    headers = headers + headers.groupby(headers).cumcount().astype(str).replace({'0':''})

    pd.DataFrame(
        data=data,
        columns=headers,
        dtype=np.uint8
    ).to_hdf(
        f"./fragments1m/balanced_fragment_{SIZE}.h5",
        key="df",
        mode="w",
        format="f"
    )


def run(overwrite_headers=False, overwrite_fragments=False, overwrite_hdfs=False):
    parse_headers(overwrite=overwrite_headers)
    split_1m(overwrite=overwrite_fragments)
    build_hdf(overwrite=overwrite_hdfs)


def main():
    run(
        overwrite_headers=False,
        overwrite_fragments=False,
        overwrite_hdfs=True
    )


if __name__ == "__main__":
    main()
