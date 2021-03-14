from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from sklearn.preprocessing import PolynomialFeatures

def linear_regression(df, target_col, cutoff_date,y_max,min_license_year,max_license_year):
    print(f"{'=' * 10}Linear Regression Method Results{'=' * 10}")

    df.reset_index(inplace=True)
    X = df[["date", "treatment"]].astype(int).values
    y = df[[target_col]].values

    lr = LinearRegression()
    lr.fit(X, y)
    effect = lr.coef_[0][1]
    print(f"Treatment effect on {target_col} is {effect}")

    plt.scatter(X[:, 0], y, c="black")
    plt.xlim([min_license_year - 0.2, max_license_year + 0.2]) # 2005.8, 2018.2
    plt.ylim([0,y_max])
    X0 = df[df["treatment"] == 0][["date", "treatment"]].astype(int).values
    X0 = np.concatenate([X0, [[cutoff_date,0]]])
    X1 = df[df["treatment"] == 1][["date", "treatment"]].astype(int).values
    X1 = np.concatenate([[[cutoff_date,1]], X1])
    plt.plot(X0[:, 0], lr.predict(X0), label="old accompaniment program")
    plt.plot(X1[:, 0], lr.predict(X1), label="new accompaniment program")
    plt.axvline(x=cutoff_date, linestyle='--', c="black", label="cut-off date")
    plt.xlabel("Year of issued license")
    if target_col == "normalized_number_of_drivers_in_accidents":
        ylabel = "number of drivers in accidents in 2019 per 10K drivers"
    else:
        ylabel = "number of drivers in accidents in 2019"
    plt.ylabel(ylabel)
    plt.title("RD by Linear Regression")
    plt.legend()
    plt.savefig(f"results/LinerRegression_{target_col}.png")
    plt.show()

    return effect

def polynomial_regression(df, target_col, cutoff_date, y_max, min_license_year,max_license_year, degree=5):
    print(f"{'=' * 10}Polynomial Regression {degree} Method Results{'=' * 10}")

    df.reset_index(inplace=True)
    X = df[["date", "treatment"]].astype(int).values
    y = df[[target_col]].values

    polyfeatures = PolynomialFeatures(degree, include_bias=False).fit_transform(X[:, 0].reshape(-1, 1))

    X_c = np.concatenate([polyfeatures, X[:, 1].reshape(-1, 1)], axis=1)

    polyreg = LinearRegression()
    polyreg.fit(X_c, y)

    effect = polyreg.coef_[0][-1]
    print(f"Treatment effect on {target_col} is {effect}")

    plt.scatter(X[:, 0], y, c="black")
    plt.xlim([min_license_year - 0.2, max_license_year + 0.2])  # 2005.8, 2018.2
    plt.ylim([0,y_max])
    X0 = X_c[X_c[:, -1] == 0]
    cutoff_date_polyfeatures = PolynomialFeatures(degree, include_bias=False).fit_transform(np.array([cutoff_date]).reshape(-1,1))
    X0 = np.concatenate([X0, [list(cutoff_date_polyfeatures[0])+[0]]])
    X1 = X_c[X_c[:, -1] == 1]
    X1 = np.concatenate([[list(cutoff_date_polyfeatures[0]) + [1]], X1])

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(X0[:, 0].min(), X0[:, 0].max(), 300)
    spl = make_interp_spline(X0[:, 0], polyreg.predict(X0), k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, label="old accompaniment program")
    xnew = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 300)
    spl = make_interp_spline(X1[:, 0], polyreg.predict(X1), k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew, power_smooth, label="new accompaniment program")

    # plt.plot(X0[:, 0], polyreg.predict(X0), c="blue", label="old accompaniment program")
    # plt.plot(X1[:, 0], polyreg.predict(X1), c="orange", label="new accompaniment program")
    plt.axvline(x=cutoff_date, linestyle='--', c="black", label="cut-off date")
    plt.xlabel("Year of issued license")
    if target_col == "normalized_number_of_drivers_in_accidents":
        ylabel = "number of drivers in accidents in 2019 per 10K drivers"
    else:
        ylabel = "number of drivers in accidents in 2019"
    plt.ylabel(ylabel)
    plt.ylabel(ylabel)
    plt.title(f"RD by Polynomial Regression w/ degree {degree}")
    plt.legend()
    plt.savefig(f"results/PolynomialRegression_deg_{degree}_{target_col}.png")
    plt.show()

    return effect

def generalization_regression(df, target_col, cutoff_date, y_max, min_license_year,max_license_year, degree=3):
    print(f"{'=' * 10}Generalization Polynomial Regression {degree} Method Results{'=' * 10}")

    df.reset_index(inplace=True)
    df['date'] = df['date'].astype(float)
    df['date'] -= cutoff_date # center
    X = df[["date", "treatment"]].astype(float).values
    y = df[[target_col]].values

    DX0 = X[:, 0]*X[:, 1]
    X0_DX0 = X[:, 0] - X[:, 0]*X[:, 1]
    polyfeatures_DX = PolynomialFeatures(degree, include_bias=False).fit_transform(DX0.reshape(-1, 1))
    polyfeaturesX_DX = PolynomialFeatures(degree, include_bias=False).fit_transform(X0_DX0.reshape(-1, 1))

    X_c = np.concatenate([polyfeatures_DX, polyfeaturesX_DX, X[:, 1].reshape(-1, 1)], axis=1)

    polyreg = LinearRegression()
    polyreg.fit(X_c, y)

    effect = polyreg.coef_[0][-1]
    print(f"Treatment effect on {target_col} is {effect}")

    plt.scatter(X[:, 0]+cutoff_date, y, c="black")
    plt.xlim([min_license_year - 0.2, max_license_year + 0.2]) # 2005.8, 2018.2
    plt.ylim([0,y_max])
    X0 = X_c[X_c[:, -1] == 0]
    cutoff_date_polyfeatures_zeros = np.zeros((1,degree))#PolynomialFeatures(degree, include_bias=False).fit_transform(np.array([0]).reshape(-1, 1))
    cutoff_date_polyfeatures0 = np.concatenate([cutoff_date_polyfeatures_zeros, cutoff_date_polyfeatures_zeros, np.array([[0]])], axis=1)
    X0 = np.concatenate([X0, cutoff_date_polyfeatures0])
    X1 = X_c[X_c[:, -1] == 1]
    cutoff_date_polyfeatures1 = np.concatenate([cutoff_date_polyfeatures_zeros, cutoff_date_polyfeatures_zeros, np.array([[1]])], axis=1)
    X1 = np.concatenate([cutoff_date_polyfeatures1, X1])

    # 300 represents number of points to make between T.min and T.max
    X0_index = degree
    xnew = np.linspace(X0[:, X0_index].min(), X0[:, X0_index].max(), 300)
    spl = make_interp_spline(X0[:, X0_index], polyreg.predict(X0), k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew + cutoff_date, power_smooth, label="old accompaniment program")
    X1_index = 0
    xnew = np.linspace(X1[:, X1_index].min(), X1[:, X1_index].max(), 300)
    spl = make_interp_spline(X1[:, X1_index], polyreg.predict(X1), k=3)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew + cutoff_date, power_smooth, label="new accompaniment program")

    # plt.plot(X0[:, 0], polyreg.predict(X0), c="blue", label="old accompaniment program")
    # plt.plot(X1[:, 0], polyreg.predict(X1), c="orange", label="new accompaniment program")
    plt.axvline(x=cutoff_date, linestyle='--', c="black", label="cut-off date")
    plt.xlabel("Year of issued license")
    if target_col == "normalized_number_of_drivers_in_accidents":
        ylabel = "number of drivers in accidents in 2019 per 10K drivers"
    else:
        ylabel = "number of drivers in accidents in 2019"
    plt.ylabel(ylabel)
    plt.title(f"RD by Generalization Polynomial Regression w/ degree {degree}")
    plt.legend()
    plt.savefig(f"results/GeneralizationPolynomialRegression_deg_{degree}_{target_col}.png")
    plt.show()

    return effect

def mean_method(df, target_col):
    df_t0 = df.loc[(df['treatment'] == 0) & (df['in_delta'] == 1)]
    df_t1 = df.loc[(df['treatment'] == 1) & (df['in_delta'] == 1)]
    df_t0 = df_t0[target_col]
    df_t1 = df_t1[target_col]
    mean_0 = df_t0.mean()
    mean_1 = df_t1.mean()
    print(f"{'=' * 10}Mean Method Results{'=' * 10}")
    effect = mean_1 - mean_0
    print(f"Treatment effect on {target_col} is {mean_1 - mean_0}")
    return effect
    # result = ttest_ind(df_t0.values, df_t1.values)
    # print(f"Two-sided T test: statistic {result[0]}, pvalue {result[1]}")
    # result = ttest_ind(df_t0.values, df_t1.values, alternative="greater")
    # print(f"One-sided T test: statistic {result[0]}, pvalue {result[1]}")

def local_linear_regression(df, target_col, delta, y_max, min_license_year,max_license_year, cutoff_date=2013):
    print(f"{'=' * 10}Local Linear Regression Method Results{'=' * 10}")

    df.reset_index(inplace=True)
    df['date'] = df['date'].astype(float)
    df['date'] -= cutoff_date  # center
    df = df[df['in_delta'] == 1]

    X = df[["date", "treatment"]].astype(float).values
    y = df[[target_col]].values
    sample_weight = 1 - np.abs(df['date'])/delta

    t0_indices = X[:,1] == 0
    t1_indices = X[:,1] == 1
    lr0 = LinearRegression()
    lr0.fit(X[t0_indices,0].reshape(-1,1), y[t0_indices], sample_weight[t0_indices])
    lr1 = LinearRegression()
    lr1.fit(X[t1_indices,0].reshape(-1,1), y[t1_indices], sample_weight[t1_indices])

    effect = lr1.intercept_[0]-lr0.intercept_[0]
    print(f"Treatment effect on {target_col} is {effect}")

    plt.scatter(X[:, 0] + cutoff_date, y, c="black")
    plt.xlim([min_license_year - 0.2, max_license_year + 0.2]) # 2005.8, 2018.2
    plt.ylim([0,y_max])
    X0 = df[df["treatment"] == 0][["date", "treatment"]].astype(float).values
    X0 = np.concatenate([X0, [[0, 0]]])
    X1 = df[df["treatment"] == 1][["date", "treatment"]].astype(float).values
    X1 = np.concatenate([[[0, 1]], X1])

    plt.plot(X0[:, 0] + cutoff_date, lr0.predict(X0[:, 0].reshape(-1,1)), label="old accompaniment program")
    plt.plot(X1[:, 0] + cutoff_date, lr1.predict(X1[:, 0].reshape(-1,1)), label="new accompaniment program")
    plt.axvline(x=cutoff_date, linestyle='--', c="black", label="cut-off date")
    plt.xlabel("Year of issued license")
    if target_col == "normalized_number_of_drivers_in_accidents":
        ylabel = "number of drivers in accidents in 2019 per 10K drivers"
    else:
        ylabel = "number of drivers in accidents in 2019"
    plt.ylabel(ylabel)
    plt.title("RD by Local Linear Regression")
    plt.legend()
    plt.savefig(f"results/LocalLinerRegression_{target_col}.png")
    plt.show()

    return effect

def local_polynomial_regression(df, target_col, delta, y_max, min_license_year,max_license_year, degree=3, cutoff_date=2013):
    print(f"{'=' * 10}Local Polynomial Regression Method Results{'=' * 10}")

    df.reset_index(inplace=True)
    df['date'] = df['date'].astype(float)
    df['date'] -= cutoff_date  # center
    df = df[df['in_delta'] == 1]

    X = df[["date", "treatment"]].astype(float).values
    y = df[[target_col]].values
    sample_weight = 1 - np.abs(df['date']) / delta

    t0_indices = X[:,1] == 0
    t1_indices = X[:,1] == 1

    polyfeatures0 = PolynomialFeatures(degree, include_bias=False).fit_transform(X[t0_indices,0].reshape(-1,1))
    polyfeatures1 = PolynomialFeatures(degree, include_bias=False).fit_transform(X[t1_indices,0].reshape(-1, 1))

    polyreg0 = LinearRegression()
    polyreg0.fit(polyfeatures0, y[t0_indices], sample_weight[t0_indices])
    polyreg1 = LinearRegression()
    polyreg1.fit(polyfeatures1, y[t1_indices], sample_weight[t1_indices])

    effect = polyreg1.intercept_[0]-polyreg0.intercept_[0]
    print(f"Treatment effect on {target_col} is {effect}")

    plt.scatter(X[:, 0] + cutoff_date, y, c="black")
    plt.xlim([min_license_year - 0.2, max_license_year + 0.2])  # 2005.8, 2018.2
    plt.ylim([0, y_max])
    cutoff_date_polyfeatures = np.zeros((1,degree))
    X0 = np.concatenate([polyfeatures0, cutoff_date_polyfeatures])
    X1 = np.concatenate([cutoff_date_polyfeatures, polyfeatures1])

    k = 3 if delta >= 3 else 1

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(X0[:, 0].min(), X0[:, 0].max(), 300)
    spl = make_interp_spline(X0[:, 0], polyreg0.predict(X0), k=k)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew + cutoff_date, power_smooth, label="old accompaniment program")
    # xnew = np.linspace(X0.min(axis=0), X0.max(axis=0), 5)
    # spl = make_interp_spline(xnew[:, 0], polyreg0.predict(xnew).squeeze(1), k=3)  # type: BSpline
    # power_smooth = spl(xnew[:,0])
    # plt.plot(xnew[:,0] + cutoff_date, power_smooth, label="old accompaniment program")

    xnew = np.linspace(X1[:, 0].min(), X1[:, 0].max(), 300)
    spl = make_interp_spline(X1[:, 0], polyreg1.predict(X1), k=k)  # type: BSpline
    power_smooth = spl(xnew)
    plt.plot(xnew + cutoff_date, power_smooth, label="new accompaniment program")
    # xnew = np.linspace(X1.min(axis=0), X1.max(axis=0), 300)
    # spl = make_interp_spline(xnew[:, 0], polyreg1.predict(xnew).squeeze(1), k=3)  # type: BSpline
    # power_smooth = spl(xnew[:,0])
    # plt.plot(xnew[:, 0] + cutoff_date, power_smooth, label="new accompaniment program")

    # plt.plot(X0[:, 0], polyreg.predict(X0), c="blue", label="old accompaniment program")
    # plt.plot(X1[:, 0], polyreg.predict(X1), c="orange", label="new accompaniment program")
    plt.axvline(x=cutoff_date, linestyle='--', c="black", label="cut-off date")
    plt.xlabel("Year of issued license")
    if target_col == "normalized_number_of_drivers_in_accidents":
        ylabel = "number of drivers in accidents in 2019 per 10K drivers"
    else:
        ylabel = "number of drivers in accidents in 2019"
    plt.ylabel(ylabel)
    plt.title(f"RD by Polynomial Regression w/ degree {degree}")
    plt.legend()
    plt.savefig(f"results/LocalPolynomialRegression_deg_{degree}_{target_col}.png")
    plt.show()

    return effect