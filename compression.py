import pandas as pd
from parser import get_headers

from time import perf_counter

# IF "C ERROR: OUT OF MEMORY", HALF THIS VALUE UNTIL VIABLE
chunksize = 1e4
df = pd.DataFrame()
non_flags = get_headers("others")

headers = next(pd.read_csv('./MH-100K/mh_100k_dataset.csv', chunksize=1)).columns.values.tolist()
dataset = pd.read_csv(
    filepath_or_buffer='./MH-100K/mh_100k_dataset.csv',
    chunksize=chunksize,
    dtype={
        header: object if header in non_flags else bool for header in headers
    },
    low_memory=False
)
chunks = []
start = perf_counter()
start2 = perf_counter()
for c, chunk in enumerate(dataset, 1):
    chunks.append(chunk)
    print(c, perf_counter() - start2, "s, chunk size:", round(chunk.memory_usage(deep=True).sum() / 1e6, 1), "mb")
    start2 = perf_counter()
print("total", perf_counter() - start, "s. Agora concatenando...")
start = perf_counter()
df = pd.concat(chunks)
print("Concat:", perf_counter() - start, "s")
print(df.info(memory_usage='deep'))
