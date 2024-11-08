import pandas as pd

chunksize = 5e3

df_class_0 = pd.read_csv('./fragments/benign_fragment_0_filtered.csv', chunksize=chunksize, delimiter=';')
df_class_1 = pd.read_csv('./fragments/malware_fragment_filtered.csv', chunksize=chunksize, delimiter=';')

benigno = next(df_class_0)
maligno = next(df_class_1)

maligno["CLASS"] = 1
benigno["CLASS"] = 0

data = pd.concat([benigno, maligno], join="outer").fillna(0)

data = data.drop(columns=['SHA256', 'NOME', 'PACOTE', 'API_MIN', 'API'])

print(data.info(memory_usage='deep'))

columns64 = data.select_dtypes(include=['int64', 'float64'])
columns64 = columns64.drop(['vt_detection', 'VT_Malware_Deteccao', 'AZ_Malware_Deteccao'], axis=1).columns

data[columns64] = data[columns64].astype('bool')


print(data.info(memory_usage='deep'))
