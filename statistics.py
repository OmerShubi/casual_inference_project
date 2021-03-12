import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def nice_plot(df,x_column,y_column, title, save_fig, plot_type='bar', rot=45, legend=False, width=None):
    if width:
        df.plot(x=x_column, y=y_column, title=title, rot=rot, kind=plot_type, label=None, width=width)
    else:
        df.plot(x=x_column, y=y_column, title=title, rot=rot, kind=plot_type, label=None)
    if legend:
        plt.legend(legend)
    plt.savefig(f"analysis/{save_fig}.png", bbox_inches='tight')
    plt.show()


def creat_year_of_license_for_each_age_table(df, name):
    # Note if data contains zeros we take them into consideration
    array = np.array([np.array(i) for i in np.array(df.index)])
    years = array[:, 0]
    ages = array[:, 1]
    year_of_license_for_each_age = {}
    for a in ages:
        year_of_license_for_each_age[a] = year_of_license_for_each_age.get(a, 0) + 1
    start_inx = 0
    for a in year_of_license_for_each_age:
        number_of_years = year_of_license_for_each_age[a]
        end_inx = start_inx + number_of_years
        year_of_license_for_each_age[a] = years[start_inx:end_inx]
        start_inx += number_of_years

    for a in year_of_license_for_each_age:
        year_of_license_for_each_age[a] = f'{year_of_license_for_each_age[a][0]}-{year_of_license_for_each_age[a][-1]}'
    # table = pd.DataFrame(year_of_license_for_each_age.keys(),data=year_of_license_for_each_age.values(), columns=['year-of-license-issued'])
    year_of_license_for_each_age_table = pd.DataFrame(data=year_of_license_for_each_age.items(),
                                                      columns=['age-range', 'year-of-license-issued'])
    print(year_of_license_for_each_age_table)
    year_of_license_for_each_age_table.to_csv(f"analysis/year_of_license_for_each_age_table_{name}.csv")
