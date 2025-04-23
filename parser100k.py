import os
import pandas as pd


def get_headers(header_type):
    f = open(f"./headers100k/{header_type}.txt", "r")
    return {line[:-1] for line in f.readlines()}


def build_fragments(dtypes, chunk_size=1e4, overwrite=False):
    if not os.path.isdir("./fragments100k"):
        os.mkdir("./fragments100k")

    if len(os.listdir("./fragments100k")) > 0 and not overwrite:
        return

    df = pd.read_csv(
        "./MH-100K/mh_100k_dataset.csv",
        chunksize=chunk_size,
        dtype=dtypes,
        low_memory=False
    )
    
    for c, chunk in enumerate(df, 1):
        chunk.loc[chunk["vt_detection"] < 4].to_csv(f"./fragments100k/benign_fragment_{c}.csv", index=False, sep=";")
        chunk.loc[chunk["vt_detection"] >= 4].to_csv(
            "./fragments100k/malware_fragment.csv",
            mode="a+",
            index=False,
            sep=';',
            header=True if c == 1 else False
        )


def get_dtypes():
    # Headers that are not booleans (flags)
    non_flags: set = get_headers("others")

    def header_type(header):
        if header in non_flags:
            if header in {"SHA256", "NOME", "PACOTE"}:
                return 'U'  # Unicode string
            return 'b'  # Signed byte
        return 'B'  # ? = Boolean, B = Unsigned byte

    headers = next(pd.read_csv("./MH-100K/mh_100k_dataset.csv", chunksize=1)).columns.values.tolist()
    return { header: header_type(header) for header in headers }


def build_hdfs(dtypes, overwrite=False):
    if os.path.isfile('./fragments100k/fragment_1.h5') and not overwrite:
        return

    malware_fragment = pd.read_csv(
        './fragments100k/malware_fragment.csv',
        index_col=False,
        sep=';',
        dtype=dtypes,
        low_memory=False
    )

    c = 1
    for file in os.listdir("./fragments100k"):
        if file.startswith("malware") or file.startswith("fragment"):
            continue
        benign_fragment = pd.read_csv(
            f'./fragments100k/benign_fragment_{c}.csv',
            index_col=False,
            sep=';',
            dtype=dtypes,
            low_memory=False
        )

        df = pd.concat([benign_fragment, malware_fragment])

        df.to_hdf(
            f"./fragments100k/fragment_{c}.h5",
            key="df",
            mode="w",
            format="f"
        )
        c += 1


def run(overwrite_fragments=False, overwrite_hdfs=False):
    dtypes = get_dtypes()
    build_fragments(dtypes=dtypes, overwrite=overwrite_fragments)
    build_hdfs(dtypes=dtypes, overwrite=overwrite_hdfs)


def main():
    run(
        overwrite_fragments=True,
        overwrite_hdfs=True
    )


if __name__ == "__main__":
    main()
