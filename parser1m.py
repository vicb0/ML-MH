import os
import math
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


def build_hdf(overwrite=False, n=1):
    if os.path.isfile('./fragments1m/balanced_fragment.h5') and not overwrite:
        return

    data = np.load(r'.\MH-1M\data\compressed\zip-intents-permissions-opcodes-apicalls\dataset.npz', allow_pickle=True)
    headers = data['column_names'].tolist()
    vt_detection = data['metadata'][:,6]

    benign = []
    malware = []

    c = 0
    for fragment in os.scandir('./fragments1m'):
        if not fragment.name.startswith('fragment_'):
            continue

        data = np.load(f'./fragments1m/fragment_{c + 1}.npz')
        data = data['arr']

        if len(malware) < 10_000:
            malwares = data[vt_detection[c * 100_000:(c + 1) * 100_000] >= 4][:10_000 - len(malware)]
            malware.extend(malwares.tolist())

        if len(benign) < 10_000:
            benigns = data[vt_detection[c * 100_000:(c + 1) * 100_000] < 4][:10_000 - len(benign)]
            benign.extend(benigns.tolist())

        if len(benign) >= 10_000 and len(malware) >= 10_000:
            break

        c += 1

    benign.extend(malware)
    df = pd.DataFrame(
        benign,
        columns=headers,
        dtype=np.uint8
    )

    class_df = pd.DataFrame({'CLASS': [0]*10_000 + [1]*10_000})
    df = pd.concat([df, class_df], axis=1)

    s = df.columns.to_series()
    df.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

    df.to_hdf(
        f"./fragments1m/balanced_fragment.h5",
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
