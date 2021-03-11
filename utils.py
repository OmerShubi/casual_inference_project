import pandas as pd
from datetime import datetime
import os
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing import PolynomialFeatures


def parse_data(parsed_data_path):
    original_data_path = "full_data.xlsx"
    if not os.path.exists(parsed_data_path):
        print(f"{'=' * 10}Parsing data{'=' * 10}")
        header = pd.read_excel(original_data_path, skiprows=1, header=None).iloc[:2, :-1].ffill(axis=1)
        dateparse = lambda x: datetime.strptime(x, '%Y %B')
        df = pd.read_excel(original_data_path, skiprows=3, header=None, index_col=[797], parse_dates=True, date_parser=dateparse)
        df.columns = pd.MultiIndex.from_arrays(header.values)
        df.rename_axis('date', inplace=True)
        df.to_excel(parsed_data_path)
    else:
        print(f"{'=' * 10}Loading data from {parsed_data_path}{'=' * 10}")
        df = pd.read_excel(parsed_data_path, header=[0, 1], index_col=0)
    return df

def clean_data(df, too_old_age=24):
    res = []
    license_year = [col[1] for col in df.columns]
    max_age = [col[0][3:] for col in df.columns]
    for year in pd.Series(df.index).dt.year:
        year_diff = year - pd.to_numeric(license_year, errors='coerce')
        year_diff[year_diff < 0] = None
        res.append(pd.to_numeric(max_age, errors='coerce') - year_diff)
    mask = pd.DataFrame(res, index=df.index.copy())
    mask.columns = df.columns.copy()
    mask[mask > too_old_age] = None
    df_final = df[mask > 0]
    df_final.dropna(axis=0, how='all', inplace=True)
    df_final.dropna(axis=1, how='all', inplace=True)
    return df_final