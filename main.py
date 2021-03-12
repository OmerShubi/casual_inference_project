from models import local_linear_regression, mean_method, linear_regression, polynomial_regression, \
    generalization_regression, local_polynomial_regression
from statistics import nice_plot, creat_year_of_license_for_each_age_table, create_genral_statistic
from utils import parse_data, clean_data, optimal_bandwidth
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def run_methods(df, methods,nmethods, num_issued_licenses_per_year, min_license_year=2006, max_license_year=2018,\
                cutoff_license_year=2013, accident_year=2019, y_max_no_normalization = 250, y_max_normalization = 0.16):
    """

    :param df:
    :return:
    """
    # prepare data for methods

    # accidents that happened during accident_year
    target_df = df.loc[pd.date_range(start=f'{accident_year}-01-01', end=f'{accident_year}-12-31', freq="MS"), :]

    # Sum over each year of license issue
    target_df_with_sum = pd.DataFrame(target_df.sum(axis=0), columns=['num_accidents']).sort_index()

    target_year_df_with_sum = target_df_with_sum.loc[(slice(None), [str(x) for x in range(min_license_year, max_license_year + 1)]),
                              :].copy(deep=True)
    target_year_df_with_sum = target_year_df_with_sum.swaplevel()
    target_year_df_with_sum.rename_axis(['date', 'age'], inplace=True)
    grouped_df = target_year_df_with_sum.groupby('date').sum()
    grouped_df.loc[[str(x) for x in range(min_license_year, cutoff_license_year)], 'treatment'] = 0
    grouped_df.loc[[str(x) for x in range(cutoff_license_year, max_license_year + 1)], 'treatment'] = 1

    grouped_df['num_issued_licenses_per_year'] = float('inf')
    # x = pd.read_csv('num_issued_licenses_per_year_all_ages.csv', index_col=1).sort_index()
    x = num_issued_licenses_per_year
    grouped_df.loc[(grouped_df.index >= str(min_license_year)) & (grouped_df.index <= str(max_license_year)),
                   'num_issued_licenses_per_year'] = x[(x.index >= min_license_year) & (x.index <= max_license_year)].values
    grouped_df['normalized_num_accidents'] = (grouped_df['num_accidents'] / grouped_df['num_issued_licenses_per_year']) * 100

    print(f"{'=' * 10}Parametric Methods{'=' * 10}\n")

    # No normalization
    y_max = y_max_no_normalization

    # linear regression all data
    lr_effect = linear_regression(df=grouped_df.copy(),
                      target_col='num_accidents',
                      cutoff_date=cutoff_license_year - 0.5,
                      y_max=y_max,
                      min_license_year=min_license_year,
                      max_license_year=max_license_year)

    # poly regression
    pr_effect = polynomial_regression(df=grouped_df.copy(),
                          target_col='num_accidents',
                          cutoff_date=cutoff_license_year - 0.5,
                          y_max=y_max,
                          min_license_year=min_license_year,
                          max_license_year=max_license_year,
                          degree=3)

    # generalization poly regression
    gn_effect = generalization_regression(df=grouped_df.copy(),
                              target_col='num_accidents',
                              cutoff_date=cutoff_license_year - 0.5,
                              y_max=y_max,
                              min_license_year=min_license_year,
                              max_license_year=max_license_year,
                              degree=3)

    print(f"\n\n{'=' * 10}Non-Parametric Methods{'=' * 10}\n")
    # Mean method
    grouped_df_to_func = grouped_df.copy()
    grouped_df_to_func.reset_index(inplace=True)
    grouped_df_to_func['date'] = grouped_df_to_func['date'].astype(int)
    delta = optimal_bandwidth(grouped_df_to_func['num_accidents'],
                              grouped_df_to_func['date'],
                              cut=cutoff_license_year - 0.5)
    delta = int(round(delta, 0))
    print(f"Optimal bandwidth {delta} for num_accidents")
    grouped_df_delta = grouped_df.copy()
    grouped_df_delta.loc[:,'in_delta'] = 0
    grouped_df_delta.loc[[str(x) for x in range(cutoff_license_year - delta, cutoff_license_year + delta)], 'in_delta'] = 1
    mean_effect = mean_method(df=grouped_df_delta.copy(), target_col='num_accidents')

    # Local linear regression
    llr_effect = local_linear_regression(df=grouped_df_delta.copy(),
                            target_col='num_accidents',
                            cutoff_date=cutoff_license_year - 0.5,
                            y_max=y_max,
                            min_license_year=min_license_year,
                            max_license_year=max_license_year,
                            delta=delta)

    # Local poly regression
    lpr_effect = local_polynomial_regression(df=grouped_df_delta.copy(),
                                target_col='num_accidents',
                                delta=delta,
                                y_max=y_max,
                                min_license_year=min_license_year,
                                max_license_year=max_license_year,
                                degree=3,
                                cutoff_date=cutoff_license_year - 0.5)

    # Normalization
    y_max = y_max_normalization

    print(f"\n\n{'=' * 10}Normalization{'=' * 10}\n")
    nlr_effect = linear_regression(df=grouped_df.copy(),
                      target_col='normalized_num_accidents',
                      cutoff_date=cutoff_license_year - 0.5,
                      y_max=y_max,
                      min_license_year=min_license_year,
                      max_license_year=max_license_year,
                      )
    npr_effect = polynomial_regression(df=grouped_df.copy(),
                          target_col='normalized_num_accidents',
                          cutoff_date=cutoff_license_year - 0.5,
                          y_max=y_max,
                          min_license_year=min_license_year,
                          max_license_year=max_license_year,
                          degree=3)
    ngn_effect = generalization_regression(df=grouped_df.copy(),
                              target_col='normalized_num_accidents',
                              cutoff_date=cutoff_license_year - 0.5,
                              y_max=y_max,
                              min_license_year=min_license_year,
                              max_license_year=max_license_year,
                              degree=3)

    grouped_df_to_func = grouped_df.copy()
    grouped_df_to_func.reset_index(inplace=True)
    grouped_df_to_func['date'] = grouped_df_to_func['date'].astype(int)
    delta = optimal_bandwidth(grouped_df_to_func['normalized_num_accidents'],
                              grouped_df_to_func['date'],
                              cut=cutoff_license_year - 0.5)
    delta = int(round(delta, 0))
    print(f"Optimal bandwidth {delta} for normalized_num_accidents")
    grouped_df_delta = grouped_df.copy()
    grouped_df_delta.loc[:, 'in_delta'] = 0
    grouped_df_delta.loc[
        [str(x) for x in range(cutoff_license_year - delta, cutoff_license_year + delta)], 'in_delta'] = 1
    nmean_effect = mean_method(df=grouped_df_delta.copy(), target_col='normalized_num_accidents')
    nllr_effect = local_linear_regression(df=grouped_df_delta.copy(),
                            target_col='normalized_num_accidents',
                            cutoff_date=cutoff_license_year - 0.5,
                            y_max=y_max,
                            min_license_year=min_license_year,
                            max_license_year=max_license_year,
                            delta=delta)

    # Local poly regression
    nlpr_effect = local_polynomial_regression(df=grouped_df_delta.copy(),
                                target_col='normalized_num_accidents',
                                delta=delta,
                                y_max=y_max,
                                min_license_year=min_license_year,
                                max_license_year=max_license_year,
                                degree=3,
                                cutoff_date=cutoff_license_year - 0.5)

    # methods_effects = {k:eval(k) for k in methods.keys()}
    methods_effects = {}
    for k in methods.keys():
        methods_effects[k] = eval(k)
    nmethods_effects = {}
    for k in nmethods.keys():
        nmethods_effects[k] = eval(k)
    return methods_effects, nmethods_effects

def run_discontinuity_assumption_check():
    # Run methods on different age range -> see no effect.
    pass

def run_analysis(df_main, df_num_issued_licenses_per_year, df_full, min_license_year=2006, max_license_year=2018,\
                 accident_year=2019):
    """

    :return:
    """
    ############ FULL DATA ##############

    ###### Range of licnse issue year per age group
    target_df_with_sum_full = pd.DataFrame(df_full.sum(axis=0), columns=['num_accidents']).sort_index()

    target_year_df_with_sum_full = target_df_with_sum_full.loc[
                              (slice(None), [str(x) for x in range(0, 3000)]),
                              :].copy(deep=True)
    target_year_df_with_sum_full = target_year_df_with_sum_full.swaplevel()
    target_year_df_with_sum_full.rename_axis(['date', 'age'], inplace=True)

    creat_year_of_license_for_each_age_table(target_year_df_with_sum_full, name='full')

    ###### Num accidents per age group
    num_accidents_per_age=target_year_df_with_sum_full.groupby(level=1).sum()
    nice_plot(num_accidents_per_age,
              x_column=None,
              y_column='num_accidents',
              title='Num accidents per age group',
              plot_type='bar', save_fig='num_accidents_per_age_group')
    ###### Num accidents per year
    num_accidents_per_year=target_year_df_with_sum_full.groupby(level=0).sum()
    nice_plot(num_accidents_per_year,
              x_column=None,
              y_column='num_accidents',
              title='Num accidents per year',
              plot_type='line', save_fig='num_accidents_per_year')
    ############ CLEAN DATA ##############

    nice_plot(df=df_num_issued_licenses_per_year.reset_index(), x_column='year_of_issue_license', y_column='num_drivers',
              title='licensed to drive in 2019 by year licensed were issued, among young drivers', save_fig="LicensedToDrive")
    # prepare data

    # accidents that happened during accident_year
    target_df = df_main.loc[pd.date_range(start=f'{accident_year}-01-01', end=f'{accident_year}-12-31', freq="MS"), :]

    # Sum over each year of license issue
    target_df_with_sum = pd.DataFrame(target_df.sum(axis=0), columns=['num_accidents']).sort_index()

    target_year_df_with_sum = target_df_with_sum.loc[
                              (slice(None), [str(x) for x in range(min_license_year, max_license_year + 1)]),
                              :].copy(deep=True)
    target_year_df_with_sum = target_year_df_with_sum.swaplevel()
    target_year_df_with_sum.rename_axis(['date', 'age'], inplace=True)

    creat_year_of_license_for_each_age_table(target_year_df_with_sum, name='clean')

    create_genral_statistic(target_year_df_with_sum)


    pass
    # number of accidents due to range age, aggregate all months, years of the accidents, aggregate all issued license
    #

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    min_license_year = 2006
    max_license_year = 2018
    cutoff_license_year = 2013
    accident_year = 2019
    y_max_no_normalization = 250
    y_max_normalization = 0.16
    parsed_data_path = "parsed_data.xlsx"
    df = parse_data(parsed_data_path=parsed_data_path)

    clean_df = clean_data(df, too_old_age=24)
    num_issued_licenses_per_year = pd.read_excel('num_issued_licenses_per_year.xlsx',
                                                 index_col=0,
                                                 skiprows=2,
                                                 usecols=[0,1],
                                                 names=["year_of_issue_license", "num_drivers"]).sort_index()
    #
    run_analysis(clean_df, num_issued_licenses_per_year, df_full=df)
    # methods = {"lr_effect": "linear regression",
    #            "pr_effect": "polynomial regression",
    #            "gn_effect": "generalization regression",
    #            "mean_effect": "Mean",
    #            "llr_effect": "local linear regression",
    #            "lpr_effect": "local polynomial regression"}
    # nmethods = {f"n{k}": f"Normalized {v}" for k, v in methods.items()}
    # methods_effects, nmethods_effects = run_methods(clean_df.copy(), methods.copy(), nmethods.copy(), num_issued_licenses_per_year)
    # df = {'Treatment Effect': [round(v, 3) for v in methods_effects.values()],
    #       'Normalized Treatment Effect':[round(v, 3) for v in nmethods_effects.values()]}
    # df = pd.DataFrame(df,
    #                   index=[methods[k] for k in methods_effects.keys()],
    #                   columns=["Treatment Effect", "Normalized Treatment Effect"])
    # df.rename_axis('Method', inplace=True)
    # print(f"\n\n{'=' * 10}Treatment Effects{'=' * 10}")
    # print(df)
    # df.to_csv("results/results.csv")


