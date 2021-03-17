from datetime import datetime
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def parse_data(parsed_data_path,original_data_path):
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

def parse_gender_df(original_data_path, parsed_data_path):
    if not os.path.exists(parsed_data_path):
        print(f"{'=' * 10}Parsing gender data{'=' * 10}")
        header = pd.read_excel(original_data_path, skiprows=1, header=None).iloc[:2, :-1].ffill(axis=1)
        df = pd.read_excel(original_data_path, skiprows=3, header=None, index_col=[797], parse_dates=True)
        df.columns = pd.MultiIndex.from_arrays(header.values)
        df.rename_axis('gender', inplace=True)
        df.to_excel(parsed_data_path)
    else:
        print(f"{'=' * 10}Loading gender data from {parsed_data_path}{'=' * 10}")
        df = pd.read_excel(parsed_data_path, header=[0, 1], index_col=0)
    return df

def clean_gender_data(df, year=2019, too_old_age=24):
    res = []
    license_year = [col[1] for col in df.columns]
    max_age = [col[0][3:] for col in df.columns]
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

def optimal_bandwidth(Y, X, cut=0):
    '''
    DESCRIPTION:
        For a given outcome Y and running variable X, computes the optimal bandwidth
        h using a triangular kernel. For more information, see
        "OPTIMAL BANDWIDTH CHOICE FOR THE REGRESSION DISCONTINUITY ESTIMATOR",
        by Imbens and Kalyanaraman, at http://www.nber.org/papers/w14726.pdf
    INPUTS:
        Two equal length pandas Series
            Y: the outcome variable
            X: the running variable
        cut: value for the threshold of the rdd (scalar) (default is 0)

    OUTPUTS:
        Scalar optimal bandwidth value
    '''

    assert (X.shape[0] == Y.shape[0], "X and Y are not of the same length")
    assert (np.sum(pd.isnull(X)) == 0, "NaNs are present in the running variable X")
    assert (np.sum(pd.isnull(Y)) == 0, "NaNs are present in the running variable X")

    # Normalize X
    X = X - cut

    # Step 1
    h1 = 1.84 * X.std() * (X.shape[0] ** (-.2))
    Nh1neg = X[(X < 0) & (X > -h1)].shape[0]
    Nh1pos = X[(X >= 0) & (X < h1)].shape[0]
    Ybarh1neg = Y[(X < 0) & (X > -h1)].mean()
    Ybarh1pos = Y[(X >= 0) & (X < h1)].mean()
    fXc = (Nh1neg + Nh1pos) / (2 * X.shape[0] * h1)
    sig2c = (((Y[(X < 0) & (X > -h1)] - Ybarh1neg) ** 2).sum() + ((Y[(X >= 0) & (X < h1)] - Ybarh1pos) ** 2).sum()) / (
                Nh1neg + Nh1pos)

    # Step 2
    medXneg = X[X < 0].median()
    medXpos = X[X >= 0].median()
    dat_temp = pd.DataFrame({'Y': Y, 'X': X})
    dat_temp = dat_temp.loc[(dat_temp['X'] >= medXneg) & (dat_temp['X'] <= medXpos)]
    dat_temp['treat'] = 0
    dat_temp.loc[dat_temp['X'] >= 0, 'treat'] = 1
    dat_temp['X2'] = X ** 2
    dat_temp['X3'] = X ** 3
    eqn = 'Y ~ 1 + treat + X + X2 + X3'
    results = smf.ols(eqn, data=dat_temp).fit()
    m3 = 6 * results.params.loc['X3']
    h2pos = 3.56 * (X[X >= 0].shape[0] ** (-1 / 7.0)) * (sig2c / (fXc * np.max([m3 ** 2, .01]))) ** (1 / 7.0)
    h2neg = 3.56 * (X[X < 0].shape[0] ** (-1 / 7.0)) * (sig2c / (fXc * np.max([m3 ** 2, .01]))) ** (1 / 7.0)
    Yplus = Y[(X >= 0) & (X <= h2pos)]
    Xplus = X[(X >= 0) & (X <= h2pos)]
    dat_temp = pd.DataFrame({'Y': Yplus, 'X': Xplus})
    dat_temp['X2'] = X ** 2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2pos = 2 * results.params.loc['X2']
    Yneg = Y[(X < 0) & (X >= -h2neg)]
    Xneg = X[(X < 0) & (X >= -h2neg)]
    dat_temp = pd.DataFrame({'Y': Yneg, 'X': Xneg})
    dat_temp['X2'] = X ** 2
    eqn = 'Y ~ 1 + X + X2'
    results = smf.ols(eqn, data=dat_temp).fit()
    m2neg = 2 * results.params.loc['X2']

    # Step 3
    rpos = 720 * sig2c / (X[(X >= 0) & (X <= h2pos)].shape[0] * h2pos ** 4)
    rneg = 720 * sig2c / (X[(X < 0) & (X >= -h2neg)].shape[0] * h2neg ** 4)
    CK = 3.4375
    hopt = CK * (2 * sig2c / (fXc * ((m2pos - m2neg) ** 2 + (rpos + rneg)))) ** .2 * Y.shape[0] ** (-.2)

    return hopt