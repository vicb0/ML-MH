import os
import pandas as pd
from parser100k import get_headers

from time import perf_counter


def parse_headers(overwrite=False):
    if not os.path.isdir("./headers100k"):
        os.mkdir("./headers100k")

    headers = pd.read_csv("./MH-100K/mh_100k_dataset.csv", chunksize=1)
    headers = next(headers).columns

    hs = ["apicall", "intent", "permission"]

    for h in hs:
        if os.path.isfile(f"./headers100k/{h}.txt") and not overwrite:
            continue
        with open(f"./headers100k/{h}.txt", "w+") as f:
            for header in headers.to_list():
                if header.lower().startswith(h):
                    f.write(header + "\n")

    if os.path.isfile("./headers100k/others.txt") and not overwrite:
        return
    with open("./headers100k/others.txt", "w+") as f:
        for header in headers.to_list():
            headerlow = header.lower()
            if headerlow.startswith("apicall") or headerlow.startswith("permission") or headerlow.startswith("intent"):
                continue
            f.write(header + "\n")


# IF "C ERROR: OUT OF MEMORY", HALF CHUNKSIZE VALUE UNTIL VIABLE
def compress(path, to_file, chunksize=1e4, overwrite=False):
    if os.path.isfile('./dataset.h5') and not overwrite:
        return

    # Headers that are not booleans (flags)
    non_flags: set = get_headers("others")

    def header_type(header):
        if header in non_flags:
            if header in {"SHA256", "NOME", "PACOTE"}:
                return 'U'  # Unicode string
            return 'b'  # Signed byte
        return 'B'  # ? = Boolean, B = Unsigned byte

    headers = next(pd.read_csv(path, chunksize=1)).columns.values.tolist()

    # low_memory=True reads the CSV in chunks, even though we are already reading in chunks.
    # Each one of these "subchunks" needs to do type conversioning again, which takes around 1 whole second
    # per subchunk. Problem is, Pandas reads in over a hundred subchunks per chunk. By setting this parameter
    # to False, and making sure the computer has at least ~6gb of memory, this ends up speeding the reading
    # process by a lot. And by defining the dtypes, we are able to store the whole dataset in memory using 
    # only 2.4Gb.
    df = pd.read_csv(
        filepath_or_buffer=path,
        chunksize=chunksize,
        dtype={
            header: header_type(header) for header in headers
        },
        low_memory=False
    )


    # Benchmarking reading speed and memory usage.
    # P.S.: Concatenating each chunk directly to the dataframe would increase the time exponentially for
    # each concat() call. For this reason, it is recommended to add all chunks to an array, and then concatting
    # the array to the dataframe in a single method call.
    chunks = []
    start = perf_counter()  # Total time
    start2 = perf_counter()  # Time per chunk
    for c, chunk in enumerate(df, 1):
        chunks.append(chunk)
        print(c, perf_counter() - start2, "s, chunk size:", round(chunk.memory_usage(deep=True).sum() / 1e6, 1), "mb")
        start2 = perf_counter()
    print("total", perf_counter() - start, "s. Concatenating...")
    start = perf_counter()
    df = pd.concat(chunks)
    print("Concat:", perf_counter() - start, "s")
    df.info(memory_usage='deep')

    # Saves in hdf format for faster reading, and no need to run this script again in the future.
    df.to_hdf(to_file, key='df', mode='w')


def run(overwrite_dataset=False, overwrite_headers=False):
    parse_headers(overwrite=overwrite_headers)
    compress("./MH-100K/mh_100k_dataset.csv", "./dataset.h5", overwrite=overwrite_dataset)


def main():
    run(
        overwrite_headers=True,
        overwrite_dataset=True
    )


if __name__ == "__main__":
    main()
