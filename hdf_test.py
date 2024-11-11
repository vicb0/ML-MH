import pandas as pd

df = pd.read_hdf('./data.h5')

df.info(memory_usage='deep')