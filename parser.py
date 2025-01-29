import os
import sys
import pandas as pd


LOG = True


class Logger(object):
    def __init__(self):
        self.log = LOG
        self.terminal = sys.stdout
    
    def write(self, msg):
        if self.log:
            self.terminal.write(msg)

    def flush(self):
        pass
sys.stdout = Logger()


def parse_headers():
    if not os.path.isdir("./headers"):
        os.mkdir("./headers")

    headers = pd.read_csv("./MH-100K/mh_100k_dataset.csv", chunksize=1)
    headers = next(headers).columns

    hs = ["apicall", "intent", "permission"]

    for h in hs:
        if os.path.isfile(f"./headers/{h}.txt"):
            continue
        with open(f"./headers/{h}.txt", "w+") as f:
            for header in headers.to_list():
                if header.lower().startswith(h):
                    f.write(header + "\n")

    if os.path.isfile("./headers/others.txt"):
        return
    with open("./headers/others.txt", "w+") as f:
        for header in headers.to_list():
            headerlow = header.lower()
            if headerlow.startswith("apicall") or headerlow.startswith("permission") or headerlow.startswith("intent"):
                continue
            f.write(header + "\n")


def get_headers(header_type):
    f = open(f"./headers/{header_type}.txt", "r")
    return {line[:-1] for line in f.readlines()}


def build_fragments(dtypes, chunk_size=1e4):
    if not os.path.isdir("./fragments"):
        os.mkdir("./fragments")

    if len(os.listdir("./fragments")) > 0:
        return

    df = pd.read_csv(
        "./MH-100K/mh_100k_dataset.csv",
        chunksize=chunk_size,
        dtype=dtypes,
        low_memory=False
    )
    
    for c, chunk in enumerate(df, 1):
        chunk.loc[chunk["vt_detection"] < 4].to_csv(f"./fragments/benign_fragment_{c}.csv", index=False, sep=";")
        chunk.loc[chunk["vt_detection"] >= 4].to_csv(
            "./fragments/malware_fragment.csv",
            mode="a+",
            index=False,
            sep=';',
            header=True if c == 1 else False
        )


def remove_useless_columns(df):
    nunique = df.nunique()
    to_drop = nunique[nunique == 1].index
    df = df.drop(to_drop, axis=1)
    
    return df


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


def build_hdfs(dtypes):
    malware_fragment = pd.read_csv(
        './fragments/malware_fragment.csv',
        index_col=False,
        sep=';',
        dtype=dtypes,
        low_memory=False
    )

    c = 1
    for file in os.listdir("./fragments"):
        if file.startswith("malware") or file.startswith("fragment"):
            continue
        benign_fragment = pd.read_csv(
            f'./fragments/benign_fragment_{c}.csv',
            index_col=False,
            sep=';',
            dtype=dtypes,
            low_memory=False
        )

        df = pd.concat([benign_fragment, malware_fragment])
        # df = remove_useless_columns(df)
        df.to_hdf(
            f"./fragments/fragment_{c}.h5",
            key="df",
            mode="w",
            format="f"
        )
        c += 1


def main():
    parse_headers()
    dtypes = get_dtypes()
    build_fragments(dtypes=dtypes)
    build_hdfs(dtypes=dtypes)


if __name__ == "__main__":
    main()
