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


def build_fragments(chunk_size=1e4):
    if not os.path.isdir("./fragments"):
        os.mkdir("./fragments")

    if len(os.listdir("./fragments")) > 0:
        return

    df = pd.read_csv("./MH-100K/mh_100k_dataset.csv", chunksize=chunk_size)
    
    for c, chunk in enumerate(df):
        chunk.loc[chunk["vt_detection"] < 4].to_csv(f"./fragments/benign_fragment_{c}.csv", index=False, sep=";")
        chunk.loc[chunk["vt_detection"] >= 4].to_csv(
            "./fragments/malware_fragment.csv",
            mode="a+",
            index=False,
            sep=';',
            header=True if c == 0 else False
        )


def remove_useless_columns(file):
    filtered_filename = f"{file[:-4]}_filtered.csv"
    if os.path.isfile(filtered_filename):
        return

    df = pd.read_csv(file, sep=';')

    nunique = df.nunique()
    to_drop = nunique[nunique == 1].index
    df = df.drop(to_drop, axis=1)
    df.to_csv(filtered_filename, index=False, sep=";")


def main():
    parse_headers()
    build_fragments()

    for file in os.listdir("./fragments"):
        if not file.endswith("_filtered.csv"):
            remove_useless_columns(f"./fragments/{file}")


if __name__ == "__main__":
    main()
