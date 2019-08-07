# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

import os
import math
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, spearmanr
from itertools import product, combinations, izip
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score, mean_absolute_error

def verify_base_and_phosphate_orientation(df):
    df["Valid"] = ((np.sign(df["T144_N"]) == -1) & (np.sign(df["T1445"]) == 1))
    df = df.drop(df[(~df["Valid"])].index, axis=0)
    df = df.drop("Valid", axis=1)
    return df


def add_psuedorotation_angle_to_df(df, method="sd"):
    if method == "sd":
        df['Theta'], _, df['T_max'], __ = izip(*df.apply(lambda row: _calculate_pseudorotation_with_sd(row), axis=1))
    else:
        df['Theta'] = df.apply(lambda row: _calculate_pseudo_rotation(row), axis=1)
        df['T_max'] = df.apply(lambda row: _calculate_max_degree(row), axis=1)

    return df


def _calculate_pseudorotation_with_sd(row):
    _theta = [row['TAU_2'], row['TAU_3'], row['TAU_4'], row['TAU_0'], row['TAU_1']]

    sum_sin = 0.0
    sum_cos = 0.0

    for i_t, t in enumerate(_theta):
        x = 0.8 * math.pi * i_t
        sum_sin += t * math.sin(x)
        sum_cos += t * math.cos(x)

    P_deg = math.degrees(math.atan2(-sum_sin, sum_cos))

    if P_deg < 0.0:
        P_deg += 360.0

    P_rad = math.radians(P_deg)
    Tm = 0.4 * (math.cos(P_rad) * sum_cos - math.sin(P_rad) * sum_sin)

    ST = 0.0
    Thc = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i_t, t in enumerate(_theta):
        Thc[i_t] = Tm * math.cos(P_rad+(0.8 * math.pi * i_t))
        d = t - Thc[i_t]
        ST += d * d

    sd_Tm = math.sqrt(0.4 * ST / 3.0)
    sd_P = sd_Tm / math.radians(Tm)
    return P_deg, sd_P, Tm, sd_Tm

def _calculate_pseudo_rotation(row):
    t0 = row['TAU_0']
    t1 = row['TAU_1']
    t2 = row['TAU_2']
    t3 = row['TAU_3']
    t4 = row['TAU_4']

    sin36 = math.sin(math.radians(36))
    sin72 = math.sin(math.radians(72))
    pseudo_rot_numerator = (t4 + t1) - (t3 + t0)
    pseudo_rot_denominator = 2 * t2 * (sin36 + sin72)

    theta_rad = math.atan2(pseudo_rot_numerator, pseudo_rot_denominator)
    theta = math.degrees(theta_rad)

    return theta


def _calculate_max_degree(row):
    t2 = row['TAU_2']
    theta = row['Theta']

    tm = t2 / math.cos(math.radians(theta))
    return tm


def create_discretized_columns(df):
    df["Gamma_sign"] = (np.sign(df["T3455"]) == 1)
    df["Gamma_sign"] = df.loc[:, "Gamma_sign"].map({True: '+', False: '-'})
    df["Gamma_syn"] = (np.abs(df["T3455"]) <= 90)
    df["Gamma_syn"] = df.loc[:, "Gamma_syn"].map({True: 'S', False: 'A'})
    df["Gamma_clinical"] = (np.abs(df["T3455"]) >= 30) & (np.abs(df["T3455"]) < 150)
    df["Gamma_clinical"] = df.loc[:, "Gamma_clinical"].map({True: 'C', False: 'P'})

    # df["Gamma"] = df["Gamma_sign"].str.cat(df["Gamma_syn"]).str.cat(df["Gamma_clinical"])
    # df["Gamma"] = pd.Categorical(df["Gamma"], categories=["+SC", "+AP", "-AP", "-SC"])

    df["Gamma"] = "other"
    df.loc[np.abs(df["T3455"]) >= 150, 'Gamma'] = "trans"
    df.loc[(df["T3455"] >= 30) & (df["T3455"] <= 90), "Gamma"] = "gauche+"
    df.loc[(df["T3455"] <= -30) & (df["T3455"] >= -90), "Gamma"] = "gauche-"

    df["Gamma"] = pd.Categorical(df["Gamma"], categories=["trans", "gauche+", "gauche-"])
    df["Gamma_syn"] = pd.Categorical(df["Gamma_syn"], categories=["S", "A"])

    df["absGamma"] = np.abs(df["T3455"])
    df["absGammaDiff90"] = np.abs(np.abs(df["T3455"])-90)

    df["Chi_sign"] = (np.sign(df["TCHI"]) == 1)
    df["Chi_sign"] = df.loc[:, "Chi_sign"].map({True: '+', False: '-'})
    df["Chi_syn"] = (np.abs(df["TCHI"]) <= 90)
    df["Chi_syn"] = df.loc[:, "Chi_syn"].map({True: 'syn', False: 'anti'})
    df["Chi"] = pd.Categorical(df["Chi_syn"], categories=["syn", "anti"])
    df["absChi"] = np.abs(df["TCHI"])
    df["absChiDiff90"] = np.abs(np.abs(df["TCHI"]) - 90)

    # C3endo_mask = (-18 <= df['Theta']) & (df['Theta'] <= 54)
    # C2endo_mask = (126 <= df['Theta']) | (df['Theta'] <= -162)

    C3endo_mask = (0 <= df['Theta']) & (df['Theta'] <= 36)
    C2endo_mask = (144 <= df['Theta']) | (df['Theta'] <= -170)

    df["Conformation"] = "Other"
    df.loc[C3endo_mask, "Conformation"] = "C3'-endo"
    df.loc[C2endo_mask, "Conformation"] = "C2'-endo"
    df["Conformation"] = pd.Categorical(df["Conformation"], categories=["C2'-endo", "C3'-endo", "Other"])

    return df


def append_measurements_to_combined_results(combined_results, query_name, measurements_df):
    measurements_df["Sugar"] = query_name.split(" ")[0]
    measurements_df["Base"] = query_name.split(" ")[1]
    measurements_df = map_combined_columns(measurements_df)

    if combined_results is None:
        return measurements_df
    else:
        combined_results = combined_results.append(measurements_df, ignore_index=False)

    return combined_results


def visualize_and_tabularize_results(result, order, base, color_columns, correlators=[]):
    for correlator in correlators:
        result.plot_scatter(correlator, order[base])

    for column in color_columns:
        result.plot_histograms(order[base], color_column=column)
        result.plot_box(order[base], color_column=column)
        for correlator in correlators:
            result.plot_scatter(correlator, order[base], color_column=column)

    result.save_ref_codes_to_file()


def map_combined_columns(df):
    mapping_dict = {
        "C1'-N1": "C1'-N1/C1'-N9",
        "C1'-N9": "C1'-N1/C1'-N9",
        "N1-C2": "N1-C2/N9-C4",
        "N9-C4": "N1-C2/N9-C4",
        "N1-C6": "N1-C6/N9-C8",
        "N9-C8": "N1-C6/N9-C8",
        "N1-C1'-C2'": "N1-C1'-C2'/N9-C1'-C2'",
        "N9-C1'-C2'": "N1-C1'-C2'/N9-C1'-C2'",
        "N1-C1'-O4'": "N1-C1'-O4'/N9-C1'-O4'",
        "N9-C1'-O4'": "N1-C1'-O4'/N9-C1'-O4'",
        "C1'-N9-C4": "C1'-N1-C2/C1'-N9-C4",
        "C1'-N1-C2": "C1'-N1-C2/C1'-N9-C4",
        "C1'-N9-C8": "C1'-N1-C6/C1'-N9-C8",
        "C1'-N1-C6": "C1'-N1-C6/C1'-N9-C8",
        "C2-N1-C6": "C2-N1-C6/C4-N9-C8",
        "C4-N9-C8": "C2-N1-C6/C4-N9-C8",
    }
    return df.rename(columns=mapping_dict)


def make_column_categorical(df, column_name):
    df[column_name] = pd.Categorical(df[column_name], categories=df[column_name].unique())


def run_subgroup_analysis(df, subgroups, correlators, measurement_order, results_folder):
    t_test_df = pd.DataFrame(index=measurement_order)
    spearman_test_df = pd.DataFrame(index=measurement_order)
    correlation_df = pd.DataFrame(index=measurement_order)

    for correlator in correlators:
        test_name = str(correlator) + ": " + "all"
        spearman_test_df[test_name] = 0
        correlation_df[test_name] = 0

        for measurement in measurement_order:
            correlator_group = df[correlator]
            measurement_group = df[measurement]
            spearman_rank = spearmanr(correlator_group, measurement_group)
            spearman_test_df.loc[measurement, test_name] = spearman_rank.pvalue
            correlation_df.loc[measurement, test_name] = spearman_rank.correlation

    for subgroup in subgroups:
        if "-" in subgroup:
            groups = subgroup.split("-")[:-1]
            subgroup = subgroup.split("-")[-1]

            group_list = []
            for col in groups:
                group_list.append(np.unique(df.loc[:, col]))

            for group in product(*group_list):
                test_name = " ".join(map(str, group)) + ": "
                group_df = df

                for idx, col in enumerate(groups):
                    group_df = group_df[(df.loc[:, col] == group[idx])]

                run_welch_test_on_subgroups(group_df, measurement_order, subgroup, t_test_df, test_name)
                for correlator in correlators:
                    run_correlation_on_subgroups(df, correlator, measurement_order, subgroup,
                                                 spearman_test_df, correlation_df, test_name)
        else:
            run_welch_test_on_subgroups(df, measurement_order, subgroup, t_test_df, "")

            for correlator in correlators:
                run_correlation_on_subgroups(df, correlator, measurement_order, subgroup,
                                             spearman_test_df, correlation_df, "")

    t_test_df.to_csv(os.path.join(results_folder, "t_tests.csv"), index=True)
    spearman_test_df.to_csv(os.path.join(results_folder, "spearman_tests.csv"), index=True)
    correlation_df.to_csv(os.path.join(results_folder, "correlation.csv"), index=True)


def run_welch_test_on_subgroups(df, measurement_order, subgroup_column, test_df, test_name_prefix):
    for pair in combinations(df.loc[:, subgroup_column].dtype.categories, 2):
        test_name = test_name_prefix + str(pair[0]) + " vs " + str(pair[1])
        test_df[test_name] = 0

        for measurement in measurement_order:
            group_a = df[(df.loc[:, subgroup_column] == pair[0])][measurement]
            group_b = df[(df.loc[:, subgroup_column] == pair[1])][measurement]
            welch_test = ttest_ind(group_a, group_b, equal_var=False, nan_policy="omit")
            test_df.loc[measurement, test_name] = welch_test.pvalue


def run_correlation_on_subgroups(df, correlator, measurement_order, subgroup_column, spearman_test_df, correlation_df,
                                 test_name_prefix):
    for subgroup in df.loc[:, subgroup_column].dtype.categories:
        test_name = str(correlator) + ": " + test_name_prefix + str(subgroup)
        correlation_df[test_name] = 0
        spearman_test_df[test_name] = 0

        for measurement in measurement_order:
            correlator_group = df[(df.loc[:, subgroup_column] == subgroup)][correlator]
            measurement_group = df[(df.loc[:, subgroup_column] == subgroup)][measurement]
            spearman_rank = spearmanr(correlator_group, measurement_group)
            spearman_test_df.loc[measurement, test_name] = spearman_rank.pvalue
            correlation_df.loc[measurement, test_name] = spearman_rank.correlation


def create_sine_regressors(df, x_col, y_cols, period, stats_df, use_base=True):
    if not os.path.exists('sugar_results/Regressors'):
        os.makedirs('sugar_results/Regressors')

    for y_col in y_cols:
        if use_base:
            for base in ["purine", "pyrimidine"]:
                name = base + "-" + y_col.replace("/", " or ")
                x = df.loc[:, x_col][df.Base == base]
                x = x.values.reshape(-1, 1)
                y = df.loc[:, y_col][df.Base == base]

                gpr = GaussianProcessRegressor(
                    kernel=(ExpSineSquared(periodicity_bounds=(period, period)) +
                            WhiteKernel(noise_level_bounds=(1e-7, 1e7))),
                    n_restarts_optimizer=100,
                    random_state=23,
                    normalize_y=True
                )
                gpr.fit(x, y)

                stats_df = save_regressor_stats(gpr, x, y, base, y_col, stats_df)
                plot_fit(x, y, gpr, name, x_col, y_col)
                joblib.dump(gpr, "sugar_results/Regressors/" + name + ".joblib")
        else:
            name = y_col.replace("/", " or ")
            x = df.loc[:, x_col]
            x = x.values.reshape(-1, 1)
            y = df.loc[:, y_col]

            gpr = GaussianProcessRegressor(
                kernel=(ExpSineSquared(periodicity_bounds=(period, period)) +
                        WhiteKernel(noise_level_bounds=(1e-7, 1e7))),
                n_restarts_optimizer=100,
                random_state=23,
                normalize_y=True
            )
            gpr.fit(x, y)

            stats_df = save_regressor_stats(gpr, x, y, "All", y_col, stats_df)
            plot_fit(x, y, gpr, name, x_col, y_col)
            joblib.dump(gpr, "sugar_results/Regressors/" + name + ".joblib")

    return stats_df


def create_linear_regressors(df, x_col, y_cols, stats_df):
    if not os.path.exists('sugar_results/Regressors'):
        os.makedirs('sugar_results/Regressors')

    for y_col in y_cols:
        for sugar in ["ribose", "deoxyribose"]:
            for conformation in ["C2'-endo", "C3'-endo", "Other"]:
                name = sugar + "-" + conformation + "-" + y_col.replace("/", " or ")
                x = df.loc[:, x_col][df.Sugar == sugar][df.Conformation == conformation]
                x = x.values.reshape(-1, 1)
                y = df.loc[:, y_col][df.Sugar == sugar][df.Conformation == conformation]

                gpr = BayesianRidge(normalize=True)
                gpr.fit(x, y)

                stats_df = save_regressor_stats(gpr, x, y, sugar + "-" + conformation, y_col, stats_df)
                plot_fit(x, y, gpr, name, x_col, y_col, color='darkblue', range_min=25, range_max=45)
                joblib.dump(gpr, "sugar_results/regressors/" + name + ".joblib")

    return stats_df


def save_regressor_stats(regressor, x, y, group, y_col, stats_df):
    y_hat = regressor.predict(x)

    try:
        decision_func = "{0} + {1:.3f}".format(str(regressor.kernel_), regressor._y_train_mean)
    except:
        decision_func = "{0:.3f}x + {1:.3f}".format(regressor.coef_[0], regressor.intercept_)

    stats = pd.DataFrame({
        "Subgroup": [group],
        "Measurement": [y_col],
        "Coefficients": [decision_func],
        "R^2": [r2_score(y, y_hat)],
        "RMSE": [math.sqrt(mean_squared_error(y, y_hat))],
        "MAE": [mean_absolute_error(y, y_hat)],
        "MAD": [median_absolute_error(y, y_hat)]
    })

    return stats_df.append(stats)


def plot_fit(x, y, regressor, title, x_label, y_label, color='darkorange', range_min=-180, range_max=180):
    import matplotlib.pyplot as plt

    X_plot = np.linspace(range_min, range_max, 10000)[:, None]
    y_mean, y_std = regressor.predict(X_plot, return_std=True)
    size = x.shape[0]
    sd = y_std.mean()

    try:
        decision_func = "{0} + {1:.3f}".format(str(regressor.kernel_), regressor._y_train_mean)
    except:
        decision_func = "{0:.3f}x + {1:.3f}".format(regressor.coef_[0], regressor.intercept_)

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, c='k', label='data')
    plt.plot(X_plot, y_mean, color=color, lw=2, label='GPR {0} (N={1}) [sd={2:.3f}]'.format(decision_func, size, sd))
    plt.fill_between(X_plot[:, 0], y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
    plt.savefig("sugar_results/Regressors/" + title + ".png")
    plt.clf()


def gpr_smoother(data, xseq, **params):
    kwargs = params['method_args']

    regressor = GaussianProcessRegressor(**kwargs)
    X = np.atleast_2d(data['x']).T
    Xseq = np.atleast_2d(xseq).T
    regressor.fit(X, data['y'])

    data = pd.DataFrame({'x': xseq})
    if params['se']:
        y, stderr = regressor.predict(Xseq, return_std=True)
        data['y'] = y
        data['se'] = stderr
        data['ymin'], data['ymax'] = y - stderr, y + stderr
    else:
        data['y'] = regressor.predict(Xseq, return_std=True)

    return data


def bayesian_ridge_smoother(data, xseq, **params):
    kwargs = params['method_args']

    regressor = BayesianRidge(**kwargs)
    X = np.atleast_2d(data['x']).T
    Xseq = np.atleast_2d(xseq).T
    regressor.fit(X, data['y'])

    data = pd.DataFrame({'x': xseq})
    if params['se']:
        y, stderr = regressor.predict(Xseq, return_std=True)
        data['y'] = y
        data['se'] = stderr
        data['ymin'], data['ymax'] = y - stderr, y + stderr
    else:
        data['y'] = regressor.predict(Xseq, return_std=True)

    return data


family_measurement_name_mappings = {
    "ribose pyrimidine": {"C1_N1": "C1'-N1", "C1_O4": "C1'-O4'", "C1_C2": "C1'-C2'", "C2_C3": "C2'-C3'",
                          "C3_C4": "C3'-C4'", "C3_O3": "C3'-O3'", "C4_C5": "C4'-C5'", "C4_O4": "C4'-O4'",
                          "C5_O5": "C5'-O5'", "C2_O2": "C2'-O2'", "N1_C2": "N1-C2", "N1_C6": "N1-C6",

                          "C1C2C3": "C1'-C2'-C3'", "C1O4C4": "C1'-O4'-C4'","C2C1O4": "C2'-C1'-O4'",
                          "C2C3C4": "C2'-C3'-C4'", "C2C3O3": "C2'-C3'-O3'", "C3C4C5": "C3'-C4'-C5'",
                          "C3C4O4": "C3'-C4'-O4'", "C4C3O3": "C4'-C3'-O3'", "C4C5O5": "C4'-C5'-O5'",
                          "C5C4O4": "C5'-C4'-O4'", "N1C1C2": "N1-C1'-C2'", "N1C1O4": "N1-C1'-O4'",
                          "C3C2O2": "C3'-C2'-O2'", "C1C2O2": "C1'-C2'-O2'",
                          "C1N1C2": "C1'-N1-C2", "C1N1C6": "C1'-N1-C6", "C2N1C6": "C2-N1-C6",

                          "T1233": "T1233", "T1445": "T1445", "T144_N": "T144_N", "T2345": "T2345",
                          "T321_N": "T321_N", "T3345": "T3345", "T4433": "T4433", "T4455": "T4455",
                          "T4122": "T4122", "T2233": "T2233", "T221_N": "T221_N", "T4322": "T4322",
                          "TAU_0": "TAU_0", "TAU_1": "TAU_1", "TAU_2": "TAU_2", "TAU_3": "TAU_3", "TAU_4": "TAU_4",
                          "T3455": "T3455", "TCHI": "TCHI",

                          "Temperature": "Temperature",
                          },
    "deoxyribose pyrimidine": {"C1_N1": "C1'-N1", "C1_O4": "C1'-O4'", "C1_C2": "C1'-C2'", "C2_C3": "C2'-C3'",
                               "C3_C4": "C3'-C4'", "C3_O3": "C3'-O3'", "C4_C5": "C4'-C5'", "C4_O4": "C4'-O4'",
                               "C5_O5": "C5'-O5'", "N1_C2": "N1-C2", "N1_C6": "N1-C6",

                               "C1C2C3": "C1'-C2'-C3'", "C1O4C4": "C1'-O4'-C4'","C2C1O4": "C2'-C1'-O4'",
                               "C2C3C4": "C2'-C3'-C4'", "C2C3O3": "C2'-C3'-O3'", "C3C4C5": "C3'-C4'-C5'",
                               "C3C4O4": "C3'-C4'-O4'", "C4C3O3": "C4'-C3'-O3'", "C4C5O5": "C4'-C5'-O5'",
                               "C5C4O4": "C5'-C4'-O4'", "N1C1C2": "N1-C1'-C2'", "N1C1O4": "N1-C1'-O4'",
                               "C1N1C2": "C1'-N1-C2", "C1N1C6": "C1'-N1-C6", "C2N1C6": "C2-N1-C6",

                               "T1233": "T1233", "T1445": "T1445", "T144_N": "T144_N", "T2345": "T2345",
                               "T321_N": "T321_N", "T3345": "T3345", "T4433": "T4433", "T4455": "T4455",
                               "TAU_0": "TAU_0", "TAU_1": "TAU_1", "TAU_2": "TAU_2", "TAU_3": "TAU_3",
                               "TAU_4": "TAU_4",
                               "T3455": "T3455", "TCHI": "TCHI",

                               "Temperature": "Temperature",
                               },
    "ribose purine": {"C1_N1": "C1'-N9", "C1_O4": "C1'-O4'", "C1_C2": "C1'-C2'", "C2_C3": "C2'-C3'",
                      "C3_C4": "C3'-C4'", "C3_O3": "C3'-O3'", "C4_C5": "C4'-C5'", "C4_O4": "C4'-O4'",
                      "C5_O5": "C5'-O5'", "C2_O2": "C2'-O2'", "N1_C4": "N9-C4", "N1_C8": "N9-C8",

                      "C1C2C3": "C1'-C2'-C3'", "C1O4C4": "C1'-O4'-C4'", "C2C1O4": "C2'-C1'-O4'",
                      "C2C3C4": "C2'-C3'-C4'", "C2C3O3": "C2'-C3'-O3'", "C3C4C5": "C3'-C4'-C5'",
                      "C3C4O4": "C3'-C4'-O4'", "C4C3O3": "C4'-C3'-O3'", "C4C5O5": "C4'-C5'-O5'",
                      "C5C4O4": "C5'-C4'-O4'", "N1C1C2": "N9-C1'-C2'", "N1C1O4": "N9-C1'-O4'",
                      "C3C2O2": "C3'-C2'-O2'", "C1C2O2": "C1'-C2'-O2'",
                      "C1N1C4": "C1'-N9-C4", "C1N1C8": "C1'-N9-C8", "C4N1C8": "C4-N9-C8",

                      "T1233": "T1233", "T1445": "T1445", "T144_N": "T144_N", "T2345": "T2345",
                      "T321_N": "T321_N", "T3345": "T3345", "T4433": "T4433", "T4455": "T4455",
                      "T4122": "T4122", "T2233": "T2233", "T221_N": "T221_N", "T4322": "T4322",
                      "TAU_0": "TAU_0", "TAU_1": "TAU_1", "TAU_2": "TAU_2", "TAU_3": "TAU_3", "TAU_4": "TAU_4",
                      "T3455": "T3455", "TCHI": "TCHI",

                      "Temperature": "Temperature",
                      },
    "deoxyribose purine": {"C1_N1": "C1'-N9", "C1_O4": "C1'-O4'", "C1_C2": "C1'-C2'", "C2_C3": "C2'-C3'",
                           "C3_C4": "C3'-C4'", "C3_O3": "C3'-O3'", "C4_C5": "C4'-C5'", "C4_O4": "C4'-O4'",
                           "C5_O5": "C5'-O5'", "N1_C4": "N9-C4", "N1_C8": "N9-C8",

                           "C1C2C3": "C1'-C2'-C3'", "C1O4C4": "C1'-O4'-C4'", "C2C1O4": "C2'-C1'-O4'",
                           "C2C3C4": "C2'-C3'-C4'", "C2C3O3": "C2'-C3'-O3'", "C3C4C5": "C3'-C4'-C5'",
                           "C3C4O4": "C3'-C4'-O4'", "C4C3O3": "C4'-C3'-O3'", "C4C5O5": "C4'-C5'-O5'",
                           "C5C4O4": "C5'-C4'-O4'", "N1C1C2": "N9-C1'-C2'", "N1C1O4": "N9-C1'-O4'",
                           "C1N1C4": "C1'-N9-C4", "C1N1C8": "C1'-N9-C8", "C4N1C8": "C4-N9-C8",

                           "T1233": "T1233", "T1445": "T1445", "T144_N": "T144_N", "T2345": "T2345",
                           "T321_N": "T321_N", "T3345": "T3345", "T4433": "T4433", "T4455": "T4455",
                           "TAU_0": "TAU_0", "TAU_1": "TAU_1", "TAU_2": "TAU_2", "TAU_3": "TAU_3",
                           "TAU_4": "TAU_4",
                           "T3455": "T3455", "TCHI": "TCHI",

                           "Temperature": "Temperature",
                           },
}
measurement_name_mappings = {
    "deoxyribose purine": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose pyrimidine": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose purine terminal": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose purine terminal C3": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose purine terminal C5": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose pyrimidine terminal": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose pyrimidine terminal C3": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose pyrimidine terminal C5": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "ribose purine": family_measurement_name_mappings["ribose purine"],
    "ribose pyrimidine": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose purine terminal": family_measurement_name_mappings["ribose purine"],
    "ribose purine terminal C3": family_measurement_name_mappings["ribose purine"],
    "ribose purine terminal C5": family_measurement_name_mappings["ribose purine"],
    "ribose pyrimidine terminal": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose pyrimidine terminal C3": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose pyrimidine terminal C5": family_measurement_name_mappings["ribose pyrimidine"],
    "deoxyribose adenine": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose cytosine": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose cytosine-protonated": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose guanine": family_measurement_name_mappings["deoxyribose purine"],
    "deoxyribose thymine": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "deoxyribose uracil": family_measurement_name_mappings["deoxyribose pyrimidine"],
    "ribose adenine": family_measurement_name_mappings["ribose purine"],
    "ribose cytosine": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose cytosine-protonated": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose guanine": family_measurement_name_mappings["ribose purine"],
    "ribose uracil": family_measurement_name_mappings["ribose pyrimidine"],
    "ribose thymine": family_measurement_name_mappings["ribose pyrimidine"],
    "combined": family_measurement_name_mappings["ribose purine"].copy()
}
measurement_name_mappings["combined"]["C1_N1"] = "C1'-N1/C1'-N9"
measurement_name_mappings["combined"]["N1_C2"] = "N1-C2/N9-C4"
measurement_name_mappings["combined"]["N1_C4"] = "N1-C2/N9-C4"
measurement_name_mappings["combined"]["N1_C6"] = "N1-C6/N9-C8"
measurement_name_mappings["combined"]["N1_C8"] = "N1-C6/N9-C8"
measurement_name_mappings["combined"]["N1C1C2"] = "N1-C1'-C2'/N9-C1'-C2'"
measurement_name_mappings["combined"]["N1C1O4"] = "N1-C1'-O4'/N9-C1'-O4'"
measurement_name_mappings["combined"]["C1N1C2"] = "C1'-N1-C2/C1'-N9-C4"
measurement_name_mappings["combined"]["C1N1C4"] = "C1'-N1-C2/C1'-N9-C4"
measurement_name_mappings["combined"]["C1N1C6"] = "C1'-N1-C6/C1'-N9-C8"
measurement_name_mappings["combined"]["C1N1C8"] = "C1'-N1-C6/C1'-N9-C8"
measurement_name_mappings["combined"]["C4N1C8"] = "C2-N1-C6/C4-N9-C8"
measurement_name_mappings["combined"]["C2N1C6"] = "C2-N1-C6/C4-N9-C8"

family_measurements_order = {
    "ribose pyrimidine": [
        "C1'-C2'", "C2'-C3'", "C3'-C4'", "C4'-O4'", "C1'-O4'", "C3'-O3'", "C4'-C5'", "C2'-O2'", "C1'-N1",
        "C5'-O5'", "N1-C2", "N1-C6",

        "C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'", "C2'-C1'-O4'", "C1'-C2'-O2'", "C3'-C2'-O2'",
        "C2'-C3'-O3'", "C4'-C3'-O3'", "C3'-C4'-C5'", "C5'-C4'-O4'", "N1-C1'-O4'", "N1-C1'-C2'", "C4'-C5'-O5'",
        "C1'-N1-C2", "C1'-N1-C6", "C2-N1-C6"
    ],
    "deoxyribose pyrimidine": [
        "C1'-C2'", "C2'-C3'", "C3'-C4'", "C4'-O4'", "C1'-O4'", "C3'-O3'", "C4'-C5'", "C1'-N1",
        "C5'-O5'", "N1-C2", "N1-C6",

        "C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'", "C2'-C1'-O4'",
        "C2'-C3'-O3'", "C4'-C3'-O3'", "C3'-C4'-C5'", "C5'-C4'-O4'", "N1-C1'-O4'", "N1-C1'-C2'", "C4'-C5'-O5'",
        "C1'-N1-C2", "C1'-N1-C6", "C2-N1-C6"
    ],
    "ribose purine": [
        "C1'-C2'", "C2'-C3'", "C3'-C4'", "C4'-O4'", "C1'-O4'", "C3'-O3'", "C4'-C5'", "C2'-O2'", "C1'-N9",
        "C5'-O5'", "N9-C4", "N9-C8",

        "C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'", "C2'-C1'-O4'", "C1'-C2'-O2'", "C3'-C2'-O2'",
        "C2'-C3'-O3'", "C4'-C3'-O3'", "C3'-C4'-C5'", "C5'-C4'-O4'", "N9-C1'-O4'", "N9-C1'-C2'", "C4'-C5'-O5'",
        "C1'-N9-C4", "C1'-N9-C8", "C4-N9-C8"
    ],
    "deoxyribose purine": [
        "C1'-C2'", "C2'-C3'", "C3'-C4'", "C4'-O4'", "C1'-O4'", "C3'-O3'", "C4'-C5'", "C1'-N9",
        "C5'-O5'", "N9-C4", "N9-C8",

        "C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'", "C2'-C1'-O4'",
        "C2'-C3'-O3'", "C4'-C3'-O3'", "C3'-C4'-C5'", "C5'-C4'-O4'", "N9-C1'-O4'", "N9-C1'-C2'", "C4'-C5'-O5'",
        "C1'-N9-C4", "C1'-N9-C8", "C4-N9-C8"
    ],
    "combined": [
        "C1'-C2'", "C2'-C3'", "C3'-C4'", "C4'-O4'", "C1'-O4'", "C3'-O3'", "C4'-C5'", "C2'-O2'", "C1'-N1/C1'-N9",
        "C5'-O5'", "N1-C2/N9-C4", "N1-C6/N9-C8",

        "C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'", "C2'-C1'-O4'", "C1'-C2'-O2'", "C3'-C2'-O2'",
        "C2'-C3'-O3'", "C4'-C3'-O3'", "C3'-C4'-C5'", "C5'-C4'-O4'",
        "N1-C1'-O4'/N9-C1'-O4'", "N1-C1'-C2'/N9-C1'-C2'", "C4'-C5'-O5'",
        "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8"
    ]
}

measurements_order = {
    "deoxyribose purine": family_measurements_order["deoxyribose purine"],
    "deoxyribose pyrimidine": family_measurements_order["deoxyribose pyrimidine"],
    "ribose purine": family_measurements_order["ribose purine"],
    "ribose pyrimidine": family_measurements_order["ribose pyrimidine"],
    "deoxyribose adenine": family_measurements_order["deoxyribose purine"],
    "deoxyribose cytosine": family_measurements_order["deoxyribose pyrimidine"],
    "deoxyribose cytosine-protonated": family_measurements_order["deoxyribose pyrimidine"],
    "deoxyribose guanine": family_measurements_order["deoxyribose purine"],
    "deoxyribose thymine": family_measurements_order["deoxyribose pyrimidine"],
    "deoxyribose uracil": family_measurements_order["deoxyribose pyrimidine"],
    "ribose adenine": family_measurements_order["ribose purine"],
    "ribose cytosine": family_measurements_order["ribose pyrimidine"],
    "ribose cytosine-protonated": family_measurements_order["ribose pyrimidine"],
    "ribose guanine": family_measurements_order["ribose purine"],
    "ribose uracil": family_measurements_order["ribose pyrimidine"],
    "ribose thymine": family_measurements_order["ribose pyrimidine"],
    "combined": family_measurements_order["combined"],
    "deoxyribose phosphate": family_measurements_order["deoxyribose purine"],
    "ribose phosphate": family_measurements_order["ribose purine"],
}

########################################################################################################################

DEOXYRIBOSE_ATOM_NAMES = {
    "C1'": "C1'",
    "C1*": "C1'",
    "C2'": "C2'",
    "C2*": "C2'",
    "C3'": "C3'",
    "C3*": "C3'",
    "C4'": "C4'",
    "C4*": "C4'",
    "O4'": "O4'",
    "O4*": "O4'",

    "O3'": "O3'",
    "O3*": "O3'",

    "C5'": "C5'",
    "C5*": "C5'",
    "O5'": "O5'",
    "O5*": "O5'",
}

DEOXYRIBOSE_PURINE_ATOM_NAMES = DEOXYRIBOSE_ATOM_NAMES.copy()
DEOXYRIBOSE_PURINE_ATOM_NAMES["C4"] = "C4"
DEOXYRIBOSE_PURINE_ATOM_NAMES["C8"] = "C8"
DEOXYRIBOSE_PURINE_ATOM_NAMES["N9"] = "N9"
DEOXYRIBOSE_PURINE_ATOM_RES = {atom: 0 for atom in set(DEOXYRIBOSE_PURINE_ATOM_NAMES.values())}
DEOXYRIBOSE_PURINE_ATOM_NAMES["P"] = "P"

DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES = DEOXYRIBOSE_ATOM_NAMES.copy()
DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES["C2"] = "C2"
DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES["C6"] = "C6"
DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES["N1"] = "N1"
DEOXYRIBOSE_PYRIMIDINE_ATOM_RES = {atom: 0 for atom in set(DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES.values())}
DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES["P"] = "P"

RIBOSE_ATOM_NAMES = DEOXYRIBOSE_ATOM_NAMES.copy()
RIBOSE_ATOM_NAMES["O2'"] = "O2'"
RIBOSE_ATOM_NAMES["O2*"] = "O2'"

RIBOSE_PURINE_ATOM_NAMES = RIBOSE_ATOM_NAMES.copy()
RIBOSE_PURINE_ATOM_NAMES["C4"] = "C4"
RIBOSE_PURINE_ATOM_NAMES["C8"] = "C8"
RIBOSE_PURINE_ATOM_NAMES["N9"] = "N9"
RIBOSE_PURINE_ATOM_RES = {atom: 0 for atom in set(RIBOSE_PURINE_ATOM_NAMES.values())}
RIBOSE_PURINE_ATOM_NAMES["P"] = "P"

RIBOSE_PYRIMIDINE_ATOM_NAMES = RIBOSE_ATOM_NAMES.copy()
RIBOSE_PYRIMIDINE_ATOM_NAMES["C2"] = "C2"
RIBOSE_PYRIMIDINE_ATOM_NAMES["C6"] = "C6"
RIBOSE_PYRIMIDINE_ATOM_NAMES["N1"] = "N1"
RIBOSE_PYRIMIDINE_ATOM_RES = {atom: 0 for atom in set(RIBOSE_PYRIMIDINE_ATOM_NAMES.values())}
RIBOSE_PYRIMIDINE_ATOM_NAMES["P"] = "P"

########################################################################################################################

DEOXYRIBOSE_ATOM_PAIRS = [
    ("C1'", "C2'", 2.0, 0, 0),
    ("C2'", "C3'", 2.0, 0, 0),
    ("C3'", "C4'", 2.0, 0, 0),
    ("C4'", "O4'", 2.0, 0, 0),
    ("C1'", "O4'", 2.0, 0, 0),
    ("C3'", "O3'", 2.0, 0, 0),
    ("C4'", "C5'", 2.0, 0, 0),
    ("C5'", "O5'", 2.0, 0, 0),
]
DEOXYRIBOSE_PURINE_ATOM_PAIRS = list(DEOXYRIBOSE_ATOM_PAIRS)
DEOXYRIBOSE_PURINE_ATOM_PAIRS.extend([("C1'", "N9", 2.0, 0, 0), ])
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS = list(DEOXYRIBOSE_ATOM_PAIRS)
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS.extend([("C1'", "N1", 2.0, 0, 0)])

RIBOSE_ATOM_PAIRS = list(DEOXYRIBOSE_ATOM_PAIRS)
RIBOSE_ATOM_PAIRS.extend([("C2'", "O2'", 2.0, 0, 0)])
RIBOSE_PURINE_ATOM_PAIRS = list(RIBOSE_ATOM_PAIRS)
RIBOSE_PURINE_ATOM_PAIRS.extend([("C1'", "N9", 2.0, 0, 0)])
RIBOSE_PYRIMIDINE_ATOM_PAIRS = list(RIBOSE_ATOM_PAIRS)
RIBOSE_PYRIMIDINE_ATOM_PAIRS.extend([("C1'", "N1", 2.0, 0, 0)])

DEOXYRIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL = list(DEOXYRIBOSE_PURINE_ATOM_PAIRS)
DEOXYRIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL.extend([("O5'", "P", 2.5, 0, 0), ("O3'", "P", 2.5, 0, +1)])
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL = list(DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS)
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL.extend([("O5'", "P", 2.5, 0, 0), ("O3'", "P", 2.5, 0, +1)])
RIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL = list(RIBOSE_PURINE_ATOM_PAIRS)
RIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL.extend([("O5'", "P", 2.5, 0, 0), ("O3'", "P", 2.5, 0, +1)])
RIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL = list(RIBOSE_PYRIMIDINE_ATOM_PAIRS)
RIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL.extend([("O5'", "P", 2.5, 0, 0), ("O3'", "P", 2.5, 0, +1)])

DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3 = list(DEOXYRIBOSE_PURINE_ATOM_PAIRS)
DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3.extend([("O5'", "P", 2.5, 0, 0)])
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3 = list(DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS)
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3.extend([("O5'", "P", 2.5, 0, 0)])
RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3 = list(RIBOSE_PURINE_ATOM_PAIRS)
RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3.extend([("O5'", "P", 2.5, 0, 0)])
RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3 = list(RIBOSE_PYRIMIDINE_ATOM_PAIRS)
RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3.extend([("O5'", "P", 2.5, 0, 0)])

DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5 = list(DEOXYRIBOSE_PURINE_ATOM_PAIRS)
DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5.extend([("O3'", "P", 2.5, 0, +1)])
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5 = list(DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS)
DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5.extend([("O3'", "P", 2.5, 0, +1)])
RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5 = list(RIBOSE_PURINE_ATOM_PAIRS)
RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5.extend([("O3'", "P", 2.5, 0, 0)])
RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5 = list(RIBOSE_PYRIMIDINE_ATOM_PAIRS)
RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5.extend([("O3'", "P", 2.5, 0, +1)])

TERMINAL_C5_DISALLOWED_ATOM_PAIRS = [("O5'", "P", 2.5, 0, 0)]
TERMINAL_C3_DISALLOWED_ATOM_PAIRS = [("O3'", "P", 2.5, 0, +1)]

########################################################################################################################

condition_mapper = {
    "Conformation__C3'-endo": ["pseudorotation", "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 18, 18/4.0],
    "Conformation__C2'-endo": ["pseudorotation", "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 162, 18/4.0],
    "Conformation__Other": None,
    "Chi__syn": ["torsion", "tO4'C1'N1C2/tO4'C1'N9C4", ["O4'", "C1'", "N1/N9", "C2/C4"], 0, 90/4.0],
    "Chi__anti": ["torsion", "tO4'C1'N1C2/tO4'C1'N9C4", ["O4'", "C1'", "N1/N9", "C2/C4"], 180, 90/4.0],
    "Gamma__gauche+": ["torsion", "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 60, 35/4.0],
    "Gamma__gauche-": ["torsion", "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], -60, 35/4.0],
    "Gamma__trans": ["torsion", "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 180, 85/4.0],
    "Func": None,
}


class RestraintFileDefinition(object):
    def __init__(self, name, pdb_codes, atom_names, atom_res, required_atom_pairs, disallowed_atom_pairs=None):
        self.name = name
        self.pdb_codes = pdb_codes
        self.atom_names = atom_names
        self.atom_res = atom_res
        self.restraints = dict()
        self.required_atom_pairs = required_atom_pairs
        self.disallowed_atom_pairs = disallowed_atom_pairs

    def append_restraint(self, group, restraint_definition):
        if group in self.restraints:
            match = False

            for definition in self.restraints[group]:
                if definition["name"] == restraint_definition["name"]:
                    match = True
                    definition["restraints"].extend(restraint_definition["restraints"])

            if not match:
                self.restraints[group].append(restraint_definition)
        else:
            self.restraints[group] = [restraint_definition]


sugar_restraint_groups = {
    "Chi-Conformation": ["C1'-C2'-O2'", "C3'-C2'-O2'", "C2'-C3'-O3'"],
    "Gamma": ["C4'-C5'", "C3'-C4'-C5'", "C5'-C4'-O4'"],
    "Conformation": ["C3'-C4'", "C2'-O2'", "C2'-C1'-O4'"],
    "Base-Func[torsion_chi]": ["N1-C1'-C2'/N9-C1'-C2'", "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8", "N1-C1'-O4'/N9-C1'-O4'"],
    "All-Func[torsion_chi]": ["C1'-N1/C1'-N9", "C1'-O4'"],
    "Sugar-Conformation-Func[tau_max]": ["C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'"],
    "Chi": ["C4'-C3'-O3'"],
    "Chi-Gamma": ["C4'-C5'-O5'"],
    "All": ["C1'-C2'", "C2'-C3'"],
    "Sugar": ["C4'-O4'"]
}

sugar_restraint_group_files = [
    RestraintFileDefinition("ribose purine", ['A', 'G', 'IG'],
                            RIBOSE_PURINE_ATOM_NAMES,
                            RIBOSE_PURINE_ATOM_RES,
                            RIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL),
    RestraintFileDefinition("ribose pyrimidine", ['C', 'T', 'U', 'IC'],
                            RIBOSE_PYRIMIDINE_ATOM_NAMES,
                            RIBOSE_PYRIMIDINE_ATOM_RES,
                            RIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL),
    RestraintFileDefinition("deoxyribose purine", ['DA', 'DG'],
                            DEOXYRIBOSE_PURINE_ATOM_NAMES,
                            DEOXYRIBOSE_PURINE_ATOM_RES,
                            DEOXYRIBOSE_PURINE_ATOM_PAIRS_NON_TERMINAL),
    RestraintFileDefinition("deoxyribose pyrimidine", ['DC', 'DT', 'DU'],
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_RES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_NON_TERMINAL)
]

terminal_C3_common_restraint_groups = {
    "Chi-Conformation": ["C1'-C2'-O2'", "C3'-C2'-O2'"],
    "Gamma": ["C4'-C5'", "C3'-C4'-C5'", "C5'-C4'-O4'"],
    "Conformation": ["C3'-C4'", "C2'-O2'", "C2'-C1'-O4'"],
    "Base-Func[torsion_chi]": ["N1-C1'-C2'/N9-C1'-C2'", "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8", "N1-C1'-O4'/N9-C1'-O4'"],
    "All-Func[torsion_chi]": ["C1'-N1/C1'-N9", "C1'-O4'"],
    "Sugar-Conformation-Func[tau_max]": ["C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'"],
    "Chi-Gamma": ["C4'-C5'-O5'"],
    "All": ["C1'-C2'", "C2'-C3'"],
    "Sugar": ["C4'-O4'"]
}

terminal_C3_restraint_groups = {
    "Sugar-Conformation": ["C3'-O3'"],
    "Chi-Conformation": ["C2'-C3'-O3'"],
    "Chi": ["C4'-C3'-O3'"],
}

terminal_C5_common_restraint_groups = {
    "Chi-Conformation": ["C1'-C2'-O2'", "C3'-C2'-O2'", "C2'-C3'-O3'"],
    "Gamma": ["C4'-C5'", "C3'-C4'-C5'", "C5'-C4'-O4'"],
    "Conformation": ["C3'-C4'", "C2'-O2'", "C2'-C1'-O4'"],
    "Base-Func[torsion_chi]": ["N1-C1'-C2'/N9-C1'-C2'", "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8", "N1-C1'-O4'/N9-C1'-O4'"],
    "All-Func[torsion_chi]": ["C1'-N1/C1'-N9", "C1'-O4'"],
    "Sugar-Conformation-Func[tau_max]": ["C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'"],
    "Chi": ["C4'-C3'-O3'"],
    "All": ["C1'-C2'", "C2'-C3'"],
    "Sugar": ["C4'-O4'"]
}

terminal_C5_restraint_groups = {
    "All": ["C5'-O5'"],
    "Chi-Gamma": ["C4'-C5'-O5'"],
}

terminal_C3_restraint_group_files = [
    RestraintFileDefinition("ribose purine terminal C3", ['A', 'G', 'IG'],
                            RIBOSE_PURINE_ATOM_NAMES,
                            RIBOSE_PURINE_ATOM_RES,
                            RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3,
                            disallowed_atom_pairs=TERMINAL_C3_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("ribose pyrimidine terminal C3", ['C', 'T', 'U', 'IC'],
                            RIBOSE_PYRIMIDINE_ATOM_NAMES,
                            RIBOSE_PYRIMIDINE_ATOM_RES,
                            RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3,
                            disallowed_atom_pairs=TERMINAL_C3_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("deoxyribose purine terminal C3", ['DA', 'DG'],
                            DEOXYRIBOSE_PURINE_ATOM_NAMES,
                            DEOXYRIBOSE_PURINE_ATOM_RES,
                            DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C3,
                            disallowed_atom_pairs=TERMINAL_C3_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("deoxyribose pyrimidine terminal C3", ['DC', 'DT', 'DU'],
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_RES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C3,
                            disallowed_atom_pairs=TERMINAL_C3_DISALLOWED_ATOM_PAIRS),
]

terminal_C5_restraint_group_files = [
    RestraintFileDefinition("ribose purine terminal C5", ['A', 'G', 'IG'],
                            RIBOSE_PURINE_ATOM_NAMES,
                            RIBOSE_PURINE_ATOM_RES,
                            RIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5,
                            disallowed_atom_pairs=TERMINAL_C5_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("ribose pyrimidine terminal C5", ['C', 'T', 'U', 'IC'],
                            RIBOSE_PYRIMIDINE_ATOM_NAMES,
                            RIBOSE_PYRIMIDINE_ATOM_RES,
                            RIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5,
                            disallowed_atom_pairs=TERMINAL_C5_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("deoxyribose purine terminal C5", ['DA', 'DG'],
                            DEOXYRIBOSE_PURINE_ATOM_NAMES,
                            DEOXYRIBOSE_PURINE_ATOM_RES,
                            DEOXYRIBOSE_PURINE_ATOM_PAIRS_TERMINAL_C5,
                            disallowed_atom_pairs=TERMINAL_C5_DISALLOWED_ATOM_PAIRS),
    RestraintFileDefinition("deoxyribose pyrimidine terminal C5", ['DC', 'DT', 'DU'],
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_NAMES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_RES,
                            DEOXYRIBOSE_PYRIMIDINE_ATOM_PAIRS_TERMINAL_C5,
                            disallowed_atom_pairs=TERMINAL_C5_DISALLOWED_ATOM_PAIRS),
]
