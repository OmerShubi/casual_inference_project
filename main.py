from utils import parse_data, non_parametric_method
import pandas as pd


if __name__ == '__main__':
    parsed_data_path = "parsed_data.xlsx"
    df = parse_data(parsed_data_path=parsed_data_path)

    target_year_df = df.loc[pd.date_range(start='2020-01-01', end='2020-12-31', freq="MS"), :]
    target_year_age_df = target_year_df.loc[:, "25-29"]
    delta = 2
    target_year_age_df_t0 = target_year_age_df.loc[:, [str(x) for x in range(2012, 2012 - delta, -1)]]
    target_year_age_df_t1 = target_year_age_df.loc[:, [str(x) for x in range(2013, 2013 + delta, 1)]]
    non_parametric_method(target_year_age_df_t0, target_year_age_df_t1)
