# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

from csdquery import CsdQuery, CsdQueryConditions, CsdMultiQueryAnalysis, CsdMultiQueryAnalysisResults, \
    CsdQueryResult, Measurement
from sugar_util import *


def process_query_results(analysis_name, query_folder, max_r, no_disorder, rem_out, max_esd):
    combined_measurements_df = None

    for base in CsdQuery.get_query_names_from_folder(query_folder):
        con_query_file_path = os.path.join(query_folder, base + ".con")
        csd_query_conditions = CsdQueryConditions(max_r, no_disorder, max_esd, rem_out,
                                                  measurements=measurement_name_mappings[base].keys())
        csd_query = CsdQuery(base, con_query_file_path, measurement_name_mappings[base], csd_query_conditions,
                             analysis_name=analysis_name)
        csd_query_result = csd_query.run()
        csd_query_result.save_ref_codes_to_file()

        # Data validation and augmentation
        csd_query_result.measurements_df = verify_base_and_phosphate_orientation(csd_query_result.measurements_df)
        csd_query_result.measurements_df = add_psuedorotation_angle_to_df(csd_query_result.measurements_df)
        csd_query_result.measurements_df = create_discretized_columns(csd_query_result.measurements_df)
        csd_query_result.recalculate_shapiro_test()

        combined_measurements_df = append_measurements_to_combined_results(combined_measurements_df, base,
                                                                        csd_query_result.measurements_df)
    make_column_categorical(combined_measurements_df, "Sugar")
    make_column_categorical(combined_measurements_df, "Base")
    results_folder = analysis_name + '_results'
    combined_measurements_df.to_csv(os.path.join(results_folder, "combined_results.csv"), index=True)

    # Subgroup analysis and visualizations
    combined_query = CsdQuery("combined", None, measurement_name_mappings["combined"], None,
                              analysis_name=analysis_name)
    combined_result = CsdQueryResult(combined_query, combined_measurements_df)
    combined_result.log_counts()
    visualize_and_tabularize_results(combined_result, measurements_order, "combined",
                                     ["Conformation", "Gamma", "Chi", "Sugar", "Base"],
                                     ["TCHI", "T_max"])
    run_subgroup_analysis(combined_measurements_df,
                          subgroups=["Conformation", "Gamma", "Gamma_syn", "Chi", "Sugar", "Base", "Chi-Conformation",
                                     "Chi-Sugar", "Chi-Base", "Gamma-Conformation", "Gamma-Sugar", "Gamma-Base",
                                     "Sugar-Base", "Sugar-Conformation", "Sugar-Conformation-Base"],
                          correlators=["absChi", "TCHI", "absChiDiff90", "absGamma", "absGammaDiff90", "T3455", "T_max",
                                       "Theta"],
                          measurement_order=measurements_order["combined"],
                          results_folder=results_folder)

    return combined_measurements_df, combined_result


if __name__ == '__main__':
    # Multi-query analysis
    r_factors = [10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5]
    non_disordered = [True, False]
    esds = [0.01, 1]
    remove_outliers = [True, False]

    sugar_analysis = CsdMultiQueryAnalysis("sugar", "sugar_queries/base families", r_factors, non_disordered, esds,
                                           remove_outliers, measurement_name_mappings)
    sugar_analysis_result = sugar_analysis.run_analysis()
    sugar_analysis_result.save_to_csvs()
    # sugar_analysis_result = CsdMultiQueryAnalysisResults.read_from_csv_files("sugar")
    sugar_analysis_result.plot_mean_analysis()
    sugar_analysis_result.plot_measurement_differences(remove_outliers=Measurement.bond, no_disorder=False, max_esd=0.01)

    # Final CSD queries and data preprocessing settings
    selected_r = 8.5
    selected_no_disorder = False
    selected_rem_out = True
    selected_max_esd = 0.01

    ############
    # Sugars
    ############
    sugar_measurements, sugar_result = process_query_results("sugar", "sugar_queries/base families",
                                                                   selected_r, selected_no_disorder,
                                                                   selected_rem_out, selected_max_esd)
    sugar_result.save_subgroup_restraints([(sugar_measurements, sugar_restraint_groups, "non_terminal_")], sugar_restraint_group_files, condition_mapper)

    stats_df = pd.DataFrame()
    stats_df = create_linear_regressors(sugar_measurements, "T_max", ["C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'"], stats_df)
    stats_df = create_sine_regressors(sugar_measurements, "TCHI", ["C1'-N1/C1'-N9", "C1'-O4'"], 180, stats_df, use_base=False)
    stats_df = create_sine_regressors(sugar_measurements, "TCHI", ["N1-C1'-C2'/N9-C1'-C2'", "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8", "N1-C1'-O4'/N9-C1'-O4'"], 360, stats_df)
    stats_df.to_csv(os.path.join("sugar_results", "regressor_results.csv"), index=False)

    sugar_result.plot_box(measurements_order["combined"])
    sugar_result.plot_box(measurements_order["combined"], x_column="Conformation", color_column="Sugar")
    sugar_result.plot_box(measurements_order["combined"], x_column="Conformation", color_column="Chi")
    sugar_result.plot_box(measurements_order["combined"], x_column="Sugar", color_column="Chi")
    sugar_result.plot_box(measurements_order["combined"], x_column="Conformation", color_column="Base")
    sugar_result._plot_custom_regressions(measurements_order["combined"], gpr_smoother, bayesian_ridge_smoother)
    sugar_result.plot_box(measurements_order["combined"], x_column="Chi", color_column="Gamma")
    sugar_result.plot_box(measurements_order["combined"], x_column="Chi", color_column="Base")

    ############
    # Terminal sugars
    ############
    terminal_measurements, terminal_result = process_query_results("sugar_terminal", "sugar_queries/terminal",
                                                                   selected_r, selected_no_disorder, selected_rem_out,
                                                                   selected_max_esd)
    terminal_result.save_subgroup_restraints([(terminal_measurements, terminal_C3_restraint_groups, "terminal_"), (sugar_measurements, terminal_C3_common_restraint_groups, "common_")], terminal_C3_restraint_group_files, condition_mapper)
    terminal_result.save_subgroup_restraints([(terminal_measurements, terminal_C5_restraint_groups, "terminal_"), (sugar_measurements, terminal_C5_common_restraint_groups, "common_")], terminal_C5_restraint_group_files, condition_mapper)
    terminal_result.plot_box(measurements_order["combined"])
    terminal_result.plot_box(measurements_order["combined"], x_column="Conformation", color_column="Sugar")
    terminal_result.plot_box(measurements_order["combined"], x_column="Conformation", color_column="Chi")
    terminal_result.plot_box(measurements_order["combined"], x_column="Sugar", color_column="Chi")
    terminal_result.plot_box(measurements_order["combined"], x_column="Chi", color_column="Gamma")
