from models import local_linear_regression,mean_method, linear_regression, polynomial_regression, \
    generalization_regression
from utils import parse_data, clean_data
import pandas as pd


def run_methods(df):
    """

    :param df:
    :return:
    """
    # prepare data for methods
    min_license_year = 2006
    max_license_year = 2018
    cutoff_license_year = 2013
    accident_year = 2019

    # accidents that happened during accident_year
    target_df = df.loc[pd.date_range(start=f'{accident_year}-01-01', end=f'{accident_year}-12-31', freq="MS"), :]

    # target_year_age_df = target_year_df.loc[:, "25-29"]

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
    x = pd.read_csv('num_issued_licenses_per_year_all_ages.csv', index_col=1).sort_index()
    grouped_df.loc[(grouped_df.index >= str(min_license_year)) & (grouped_df.index <= str(max_license_year)),
                   'num_issued_licenses_per_year'] = x[(x.index >= min_license_year) & (x.index <= max_license_year)].values
    grouped_df['normalized_num_accidents'] = (grouped_df['num_accidents'] / grouped_df['num_issued_licenses_per_year']) * 100

    print(f"{'=' * 10}Parametric Methods{'=' * 10}\n")

    # linear regression all data
    linear_regression(df=grouped_df.copy(),
                      target_col='num_accidents',
                      cutoff_date=cutoff_license_year - 0.5)

    # poly regression
    polynomial_regression(df=grouped_df.copy(),
                          target_col='num_accidents',
                          cutoff_date=cutoff_license_year - 0.5,
                          degree=3)

    # generalization poly regression
    generalization_regression(df=grouped_df.copy(),
                              target_col='num_accidents',
                              cutoff_date=cutoff_license_year - 0.5,
                              degree=3)

    print(f"\n\n{'=' * 10}Non-Parametric Methods{'=' * 10}\n")
    # Mean methods
    delta = 3
    grouped_df_delta = grouped_df.copy()
    grouped_df_delta.loc[:,'in_delta'] = 0
    grouped_df_delta.loc[[str(x) for x in range(cutoff_license_year - delta, cutoff_license_year + delta)], 'in_delta'] = 1
    mean_method(df=grouped_df_delta.copy(), target_col='num_accidents')

    # Local linear regression
    local_linear_regression(df=grouped_df_delta.copy(),
                            target_col='num_accidents',
                            cutoff_date=cutoff_license_year - 0.5,
                            delta=delta)

    # Normalization
    print(f"\n\n{'=' * 10}Normalization{'=' * 10}\n")
    linear_regression(df=grouped_df.copy(),
                      target_col='normalized_num_accidents',
                      cutoff_date=cutoff_license_year - 0.5)
    polynomial_regression(df=grouped_df.copy(),
                          target_col='normalized_num_accidents',
                          cutoff_date=cutoff_license_year - 0.5,
                          degree=3)
    generalization_regression(df=grouped_df.copy(),
                              target_col='normalized_num_accidents',
                              cutoff_date=cutoff_license_year - 0.5,
                              degree=3)
    mean_method(df=grouped_df_delta.copy(), target_col='normalized_num_accidents')
    local_linear_regression(df=grouped_df_delta.copy(),
                            target_col='normalized_num_accidents',
                            cutoff_date=cutoff_license_year - 0.5,
                            delta=delta)


def run_discontinuity_assumption_check():
    # Run methods on different age range -> see no effect.
    pass

def run_analysis():
    """

    :return:
    """
    pass


if __name__ == '__main__':
    parsed_data_path = "parsed_data.xlsx"
    df = parse_data(parsed_data_path=parsed_data_path)
    clean_df = clean_data(df, too_old_age=24)
    #
    # run_analysis()
    #
    run_methods(clean_df)
