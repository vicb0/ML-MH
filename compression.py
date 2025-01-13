import pandas as pd
from parser import get_headers

from time import perf_counter


# # IF "C ERROR: OUT OF MEMORY", HALF CHUNKSIZE VALUE UNTIL VIABLE
# def compress(df, to_file, chunksize=1e4):
#     # Empty dataframe to store dataset after reading chunks and concatenating
#     new_df = pd.DataFrame()
    

#     # low_memory=True reads the CSV in chunks, even though we are already reading in chunks.
#     # Each one of these "subchunks" needs to do type conversioning again, which takes around 1 whole second
#     # per subchunk. Problem is, Pandas reads in over a hundred subchunks per chunk. By setting this parameter
#     # to False, and making sure the computer has at least ~6gb of memory, this ends up speeding the reading
#     # process by a lot. And by defining the dtypes, we are able to store the whole dataset in memory using 
#     # only 2.4Gb.
#     if type(df) == str:
#         df = pd.read_csv(
#             filepath_or_buffer=df,
#             chunksize=chunksize,
#             dtype={
#                 header: header_type(header) for header in headers
#             },
#             low_memory=False
#         )


#     # Benchmarking reading speed and memory usage.
#     # P.S.: Concatenating each chunk directly to the dataframe would increase the time exponentially for
#     # each concat() call. For this reason, it is recommended to add all chunks to an array, and then concatting
#     # the array to the dataframe in a single method call.
#     chunks = []
#     start = perf_counter()  # Total time
#     start2 = perf_counter()  # Time per chunk
#     for c, chunk in enumerate(dataset, 1):
#         chunks.append(chunk)
#         print(c, perf_counter() - start2, "s, chunk size:", round(chunk.memory_usage(deep=True).sum() / 1e6, 1), "mb")
#         start2 = perf_counter()
#     print("total", perf_counter() - start, "s. Concatenating...")
#     start = perf_counter()
#     df = pd.concat(chunks)
#     print("Concat:", perf_counter() - start, "s")
#     df.info(memory_usage='deep')

#     # Saves in hdf format for faster reading, and no need to run this script again in the future.
#     new_df.to_hdf(to_file, key='df', mode='w')

for i in range(1, 11 + 1):
    df = pd.read_hdf(f"./fragments/fragment_{i}.h5")
    print(i)
    print(df.loc[df["vt_detection"] >= 4].shape)
    print(df.loc[df["vt_detection"] < 4].shape)
    print()