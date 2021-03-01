import pandas as pd
from datetime import datetime
import os
from scipy.stats import ttest_ind

def parse_data(parsed_data_path):
    original_data_path = "full_data.xlsx"
    if not os.path.exists(parsed_data_path):
        header = pd.read_excel(original_data_path, skiprows=1, header=None).iloc[:2, :-1].ffill(axis=1)
        dateparse = lambda x: datetime.strptime(x, '%Y %B')
        df = pd.read_excel(original_data_path, skiprows=3, header=None, index_col=[797], parse_dates=True, date_parser=dateparse)#.rename(columns={797: 'A'}).set_index('A')
        df.columns = pd.MultiIndex.from_arrays(header.values)
        df.to_excel(parsed_data_path)
    else:
        df = pd.read_excel(parsed_data_path, header=[0,1], index_col=0)
    return df

def non_parametric_method(df_t0, df_t1):
    print(f"{'='*10}Running Non Parametric Method{'='*10}")
    sum_0 = df_t0.sum(axis=0)
    sum_1 = df_t1.sum(axis=0)
    mean_0 = sum_0.mean()
    mean_1 = sum_1.mean()
    print(f"{'='*10}Non Parametric Method Results{'='*10}")
    print(f"Mean Treatment 0: {mean_0}, Mean Treatment 1: {mean_1}")
    result = ttest_ind(sum_0, sum_1)
    print(f"Two-sided T test: statistic {result[0]}, pvalue {result[1]}")
    result = ttest_ind(sum_0, sum_1, alternative = "greater")
    print(f"One-sided T test: statistic {result[0]}, pvalue {result[1]}")

