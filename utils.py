import pandas as pd
from datetime import datetime
import os
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
        df = pd.read_excel(parsed_data_path, header=[0,1], index_col=0)
    return df

def non_parametric_method(df_t0, df_t1, col_name):
    df_t0 = df_t0[col_name]
    df_t1 = df_t1[col_name]
    mean_0 = df_t0.mean()
    mean_1 = df_t1.mean()
    print(f"{'='*10}Non Parametric Method on {col_name} Results{'='*10}")
    print(f"Mean Treatment 0: {mean_0}, Mean Treatment 1: {mean_1}")
    result = ttest_ind(df_t0.values, df_t1.values)
    print(f"Two-sided T test: statistic {result[0]}, pvalue {result[1]}")
    result = ttest_ind(df_t0.values, df_t1.values, alternative = "greater")
    print(f"One-sided T test: statistic {result[0]}, pvalue {result[1]}")

def linear_regression(df_to_model, df_to_plot):
    df_to_plot.loc[df_to_plot.index < str(2013),"treatment"] = 0
    df_to_plot.loc[df_to_plot.index >= str(2013),"treatment"] = 1
    df_to_plot.drop("unknown", inplace=True)
    df_to_plot = df_to_plot[df_to_plot.index >= str("2009")]
    df_to_plot.reset_index(inplace=True)

    df_to_model.reset_index(inplace=True)
    X = df_to_model[["index", "treatment"]].astype(int).values
    y = df_to_model[["num_accidents"]].values
    print(f"{'=' * 10}Linear Regression Method Results{'=' * 10}")
    lr = LinearRegression()
    lr.fit(X,y)
    print(f"Treatment effect on num_accidents is {lr.coef_[0][1]}")

    y_normalized = df_to_model[["normalized_num_accidents"]].values
    lr_normalized = LinearRegression()
    lr_normalized.fit(X,y_normalized)
    print(f"Treatment effect on normalized_num_accidents is {lr_normalized.coef_[0][1]}")

    X_plot = df_to_plot[["index", "treatment"]].astype(int).values
    y_plot = df_to_plot[["num_accidents"]].values
    plt.scatter(X_plot[:,0], y_plot, c="black", label='out-of-delta points')
    plt.scatter(X[:,0], y, c="red", label='in-delta points')
    plt.xlim([2008.8, 2020.2])
    X0 = df_to_plot[df_to_plot["treatment"] == 0][["index", "treatment"]].astype(int).values
    X1 = df_to_plot[df_to_plot["treatment"] == 1][["index", "treatment"]].astype(int).values
    plt.plot(X0[:,0], lr.predict(X0), c="blue", label="old accompaniment program")
    plt.plot(X1[:,0], lr.predict(X1), c="orange", label="new accompaniment program")
    plt.axvline(x=2013, linestyle='--', c="black", label="cut-off year")
    plt.xlabel("Year of issued license")
    plt.ylabel("Number of accidents in 2020")
    plt.title("Regression Discontinuity by Linear Regression on 25-29 year olds")
    plt.legend()
    plt.savefig("LinerRegression.png")
    plt.show()




