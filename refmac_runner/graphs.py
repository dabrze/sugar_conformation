# coding: utf-8

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.legend_handler import HandlerLine2D

__author__ = "Marcin Kowiel, Dariusz Brzezinski"


def read_file(dir, file_name, verbose=True):
    df = pd.read_csv(os.path.join(dir, file_name), sep=";", header=0, index_col=None)
    df = df.loc[1:]

    if verbose:
        print file_name
        print df

    return df


def setup_canvas():
    sns.set(style="ticks", rc={"lines.linewidth": 2, "lines.markersize": 8}, font_scale=1.5)
    sns.set_palette(["#3498db", "#3498db", "#e74c3c", "#e74c3c", "#2ecc71", "#2ecc71", "#34495e", "#34495e"])
    fig = plt.figure(figsize=(12, 7.5))

    return fig


def create_legend(lns):
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="center right", #  bbox_to_anchor=(1.1, 0.5), borderaxespad=0.,
               handler_map={type(lns[0]): HandlerLine2D(marker_pad=-0.05, numpoints=2)})


def limit_axes(fig, x_min=None, x_max=None):
    if x_min is not None and x_max is not None:
        for i, ax in enumerate(fig.axes):
            ax.set_xlim(x_min, x_max)

    plt.subplots_adjust(left=0.1, right=0.82, top=0.95, bottom=0.11)
    

def create_title(df_external, x_axis_column):
    subtitle = ""
    if x_axis_column == "weight_matrix":
        subtitle = "external_scale={}; external_gmwt={}".format(df_external['external_distance'][1],
                                                                df_external['external_gmwt'][1])
    elif x_axis_column == "external_distance":
        subtitle = "weight_matrix={}; external_gmwt={}".format(df_external['weight_matrix'][1],
                                                               df_external['external_gmwt'][1])
    elif x_axis_column == "external_gmwt":
        subtitle = "weight_matrix={}; external_scale={}".format(df_external['weight_matrix'][1],
                                                                df_external['external_distance'][1])
    else:
        subtitle = "weight_matrix={}; external_scale={}; external_gmwt={}".format(df_external['weight_matrix'][1],
                                                                                  df_external['external_distance'][1],
                                                                                  df_external['external_gmwt'][1])

    plt.title('2han\n' + subtitle)


def plot_parameter_sensitivity(log_dir, log_file, log_external_file, x_axis_column, x_axis_title, filename,
                               fixed_reference=False, x_min=None, x_max=None, y_min=None, y_max=None, y2_min=None,
                               y2_max=None, y3_min=None, y3_max=None, every=1):
    df = read_file(log_dir, log_file)
    df_external = read_file(log_dir, log_external_file)
    fig = setup_canvas()

    if fixed_reference:
        reference_row = df[df["weight_matrix"] == df_external["weight_matrix"][1]].iloc[0].copy().squeeze()
        df = df.drop(df.index[range(df.shape[0])])

        for i in range(df_external.shape[0]):
            df.loc[i] = reference_row
            df.loc[i, x_axis_column] = df_external.loc[i+1, x_axis_column]

    ax = fig.add_subplot(111)
    ax.set_ylabel('$R$ factor [%]')
    ax.set_xlabel(x_axis_title)

    r = ax.plot(df[x_axis_column], 100.0*df['r_factor_final'], label='$R$ factor', linestyle='--', marker="s", markevery=every)
    r_ext = ax.plot(df_external[x_axis_column], 100.0*df_external['r_factor_final'], label='$R$ factor - external', marker="s", markevery=every)
    r_free = ax.plot(df[x_axis_column], 100.0*df['r_free_final'], label='$R_{free}$', linestyle='--', marker="o", markevery=every)
    r_free_ext = ax.plot(df_external[x_axis_column], 100.0*df_external['r_free_final'], label='$R_{free}$ - external', marker="o", markevery=every)

    ax2 = ax.twinx()
    ax2.set_ylabel('$RMSD$ (bond) [$\AA$]')

    rms_bond = ax2.plot(df[x_axis_column], df['rms_bond_lenght_final'], label='$RMSD$ (bond)', linestyle='--', marker="^", markevery=every, color=sns.color_palette()[4])
    rms_bond_ext = ax2.plot(df_external[x_axis_column], df_external['rms_bond_lenght_final'], label='$RMSD$ (bond) - external', marker="^", markevery=every, color=sns.color_palette()[5])

    ax3 = ax.twinx()
    ax3.set_ylabel('$RMSD$ (angle) [$^\circ$]')
    ax3.spines['right'].set_position(('axes', 1.16))

    rms_angle = ax3.plot(df[x_axis_column], df['rms_bond_angle_final'], label='$RMSD$ (angle)', linestyle='--', marker="d", markevery=every, color=sns.color_palette()[6])
    rms_angle_ext = ax3.plot(df_external[x_axis_column], df_external['rms_bond_angle_final'], label='$RMSD$ (angle) - external', marker="d", markevery=every, color=sns.color_palette()[7])

    limit_axes(fig, x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(y2_min, y2_max)
    ax3.set_ylim(y3_min, y3_max)

    create_legend(r + r_ext + r_free + r_free_ext + rms_bond + rms_bond_ext + rms_angle + rms_angle_ext)

    plt.savefig(filename + ".svg", format="svg", bbox_inches="tight")
    #create_title(df_external, x_axis_column)
    plt.savefig(filename, dpi=300, bbox_inches="tight")


plot_parameter_sensitivity("sugar_results", "run_standard_2han.log", "run_external_2han.log",
                           x_axis_column="weight_matrix", x_axis_title="$w$",
                           filename="sugar_results/2han_weight_matrix.png", x_min=0.01, x_max=0.4,
                           y_min=16, y_max=23, y2_min=0.005, y2_max=0.030, y3_min=1.5, y3_max=3.0, every=2)

plot_parameter_sensitivity("sugar_results", "run_standard_2han.log", "run_external_scale_2han.log",
                           x_axis_column="external_distance", x_axis_title="$w_{ext}$",
                           filename="sugar_results/2han_external_distance.png", fixed_reference=True, x_min=0.1, x_max=4,
                           y_min=17.5, y_max=22, y2_min=0.005, y2_max=0.015, y3_min=1.5, y3_max=2.0, every=2)