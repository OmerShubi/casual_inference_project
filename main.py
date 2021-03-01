from utils import parse_data, non_parametric_method, linear_regression
import pandas as pd

def run_methods():
    # prepare data for methods
    target_year_df = df.loc[pd.date_range(start='2020-01-01', end='2020-12-31', freq="MS"), :]
    target_year_age_df = target_year_df.loc[:, "25-29"]

    target_year_age_df_with_sum = pd.DataFrame(target_year_age_df.sum(axis=0), columns=['num_accidents'])

    delta = 2
    target_year_age_df_t0 = target_year_age_df_with_sum.loc[[str(x) for x in sorted(range(2012, 2012 - delta, -1))]]
    target_year_age_df_t1 = target_year_age_df_with_sum.loc[[str(x) for x in range(2013, 2013 + delta, 1)]]

    target_year_age_df_t0['treatment'] = 0
    target_year_age_df_t1['treatment'] = 1

    # Method 1
    non_parametric_method(target_year_age_df_t0["num_accidents"], target_year_age_df_t1["num_accidents"])

    # Method 2
    target_year_age_df_with_indicator = pd.concat([target_year_age_df_t0, target_year_age_df_t1])
    linear_regression(target_year_age_df_with_indicator, target_year_age_df_with_sum)

def run_analysis():
    pass


if __name__ == '__main__':
    parsed_data_path = "parsed_data.xlsx"
    df = parse_data(parsed_data_path=parsed_data_path)

    run_analysis()

    run_methods()

