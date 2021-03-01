import pandas as pd

fname = "full_data.xlsx"
header = pd.read_excel(fname, skiprows=1, header=None).iloc[:2, 2:].ffill(axis=1)
df = pd.read_excel(fname, skiprows=3, header=None).rename(columns={889: 'A', 890: 'B'}).set_index(['A', 'B'])
df.columns = pd.MultiIndex.from_arrays(header.values)
pass