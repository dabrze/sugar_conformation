# coding: utf-8
# Authors: Dariusz Brzezinski <dariusz.brzezinski@cs.put.poznan.pl>

import os
import json
import pprint
import glob
import itertools
import scipy.stats
import numpy as np
import pandas as pd
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enum import Enum
import logging
import ccdc.search
import ccdc.io
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from plotnine import ggplot, aes, geom_histogram, facet_wrap, theme, theme_bw, ggsave, geom_freqpoly, element_text, \
    scale_x_continuous, geom_point, geom_line, facet_grid, scale_color_manual, element_blank, labs, scale_fill_manual, \
    geom_density, geom_bar, scale_color_brewer, scale_color_distiller, geom_vline, geom_text, ggtitle, geom_smooth, \
    geom_boxplot, geom_violin, stat_summary, position_dodge, scale_y_continuous, scale_linetype_manual
from mizani.breaks import mpl_breaks
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

logging.basicConfig(level=logging.INFO)

PLOT_DPI = 300


class Measurement(Enum):
    other = 0
    bond = 1
    angle = 2
    torsion = 3


class CsdQueryConditions(object):
    def __init__(self, max_r_factor=50, no_disorder=False, max_esd=1, remove_outliers=False, measurements=None):
        self.max_r_factor = max_r_factor
        self.no_disorder = no_disorder
        self.max_esd = max_esd
        self.remove_outliers = remove_outliers
        self.measurements = measurements

    def __str__(self):
        return "max_r_factor=%s, no_disorder=%s, max_esd=%s, remove_outliers=%s" % (self.max_r_factor, self.no_disorder,
                                                                                    self.max_esd, self.remove_outliers)


class CsdAnalysisComponent(object):
    def __init__(self, analysis_name):
        self.analysis_name = analysis_name

    def _create_results_save_path(self, result_file_name, subfolder=None):
        results_folder = self.analysis_name + '_results'

        if subfolder is not None:
            results_folder = os.path.join(results_folder, subfolder)

        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        return os.path.join(results_folder, result_file_name)

    def _save_plot(self, plot, plot_name, width, height, dpi, subfolder=None):
        import warnings
        warnings.filterwarnings("ignore")

        plot_file = self._create_results_save_path(plot_name, subfolder)
        logging.info("Saving plot to: %s", plot_file)
        ggsave(plot=plot, dpi=PLOT_DPI, width=width, height=height, filename=plot_file)

        svg_plot_file = plot_file[:-4] + ".svg"
        logging.info("Saving plot to: %s", svg_plot_file)
        ggsave(plot=plot, dpi=PLOT_DPI, width=width, height=height, filename=svg_plot_file)


class CsdQuery(CsdAnalysisComponent):
    TEMP_COLUMN = "Temperature"

    def __init__(self, name, con_file_path, measurement_name_mapping=None, query_conditions=None, analysis_name=None):
        self.base = super(CsdQuery, self)
        self.base.__init__(analysis_name)
        self.name = name
        self.con_file_path = con_file_path
        self.measurement_name_mapping = measurement_name_mapping
        self.conditions = query_conditions if query_conditions is not None else CsdQueryConditions()

    def run(self):
        logging.info("Running query: %s (%s)", self.name, self.conditions)

        hits = self._run_csd_search()
        hits = self._filter_query_hits_for_max_esd(hits)
        measurements_df = self._convert_hits_to_measurements_data_frame(hits)
        measurements_df = self._add_temperature_to_measurements(measurements_df, hits)
        measurements_df = self._leave_only_selected_measurements(measurements_df)

        if self.conditions.remove_outliers:
            if isinstance(self.conditions.remove_outliers, Measurement):
                measurements_df = self._remove_outliers(measurements_df, type=self.conditions.remove_outliers)
            else:
                measurements_df = self._remove_outliers(measurements_df)

        if self.measurement_name_mapping is not None:
            measurements_df = measurements_df.rename(columns=self.measurement_name_mapping)

        return CsdQueryResult(self, measurements_df)

    def _run_csd_search(self):
        con_substructure = ccdc.search.ConnserSubstructure(self.con_file_path)
        substructure_search = ccdc.search.SubstructureSearch()
        substructure_search.add_substructure(con_substructure)
        substructure_search.settings.max_r_factor = self.conditions.max_r_factor
        substructure_search.settings.no_disorder = self.conditions.no_disorder
        hits = substructure_search.search()

        return hits

    def _filter_query_hits_for_max_esd(self, query_hits):
        selected_hits = []
        for hit in query_hits:
            if hit.entry._entry.experimental_info().mean_cc_bond_sigma().maximum_value() <= self.conditions.max_esd:
                selected_hits.append(hit)

        return selected_hits

    def _get_bond_names(self):
        return [k for k in self.measurement_name_mapping.keys() if '_' in k and 'T' not in k and len(k) <= 6]

    def _get_angle_names(self):
        return [k for k in self.measurement_name_mapping.keys() if '_' not in k and 'T' not in k and len(k) <= 6]

    def _get_torsion_names(self):
        return [k for k in self.measurement_name_mapping.keys() if 'T' in k and len(k) <= 6]

    def _get_mapped_bond_names(self):
        return [self.measurement_name_mapping[m] for m in self._get_bond_names()]

    def _get_mapped_angle_names(self):
        return [self.measurement_name_mapping[m] for m in self._get_angle_names()]

    def _get_mapped_torsion_names(self):
        return [self.measurement_name_mapping[m] for m in self._get_torsion_names()]

    @classmethod
    def _convert_hits_to_measurements_data_frame(cls, hits):
        return pd.DataFrame([h.measurements for h in hits], index=cls._index_hits(hits))

    @classmethod
    def _index_hits(cls, hits):
        return list([h.identifier + "_" + str(idx) for idx, h in enumerate(hits)])

    @classmethod
    def _add_temperature_to_measurements(cls, measurements_df, hits):
        measurements_df.loc[:, cls.TEMP_COLUMN] = pd.Series(
            [h.entry._entry.experimental_info().temperature() for h in hits],
            index=cls._index_hits(hits))
        return measurements_df

    def _leave_only_selected_measurements(self, measurements_df):
        if self.conditions.measurements is not None:
            return measurements_df.loc[:, self.conditions.measurements]
        else:
            return measurements_df

    def _remove_outliers(self, measurements_df, z_threshold=3.5, type=None):
        if self.TEMP_COLUMN in measurements_df.columns:
            test_df = measurements_df.drop(self.TEMP_COLUMN, axis=1)
        else:
            test_df = measurements_df

        if type is not None:
            test_df = self._leave_only_one_measurment_type(test_df, [type])
        else:
            test_df = self._leave_only_one_measurment_type(test_df, [Measurement.bond, Measurement.angle])

        median = test_df.median()
        mad = test_df.subtract(median).abs().median()
        z_test = (0.6745 * test_df.subtract(median) / mad).abs() < z_threshold
        measurements_without_outliers = measurements_df.loc[z_test.all(axis=1), :]

        logging.info("Before outlier removal: " + str(measurements_df.shape[0]) +
                     " After MAD outlier removal: " + str(measurements_without_outliers.shape[0]))

        return measurements_without_outliers

    def _leave_only_one_measurment_type(self, df, types):
        cols = []

        for type in types:
            if type == Measurement.bond:
                cols.extend(self._get_bond_names())
            elif type == Measurement.angle:
                cols.extend(self._get_angle_names())
            elif type == Measurement.torsion:
                cols.extend(self._get_angle_names())

        return df.loc[:, cols]

    @staticmethod
    def get_query_names_from_folder(query_folder):
        query_names = []

        for file_path in glob.glob(os.path.join(query_folder, "*.con")):
            file_name = os.path.basename(file_path)
            query_names.append(os.path.splitext(file_name)[0])

        return query_names


class CsdQueryResult(CsdAnalysisComponent):
    NAME_COLUMN = "Name"
    MAX_R_FACTOR_COLUMN = "Max R-factor"
    MAX_ESD_COLUMN = "Max e.s.d. (C-C)"
    BOND_ANGLE_STD_MEAN_COLUMN = "Bond angle std mean"
    BOND_LENGTH_STD_MEAN_COLUMN = "Bond length std mean"
    BOND_ANGLE_SEM_COLUMN = "Avg. bond angle SEM "
    BOND_LENGTH_SEM_COLUMN = "Avg. bond length SEM"
    REMOVE_OUTLIERS_COLUMN = "Remove outliers"
    STRUCTURE_COUNT_COLUMN = "Structure count"
    NO_DISORDER_COLUMN = "No disorder"
    STANDARD_DEVIATION_COLUMN = "Standard deviation"
    MEASUREMENT_COLUMN = "Measurement"
    MEAN_COLUMN = "Mean"
    FILE_COUNT_COLUMN = "File count"
    VALUE_COLUMN = "Value"
    SHAPIRO_TEST_COLUMN = "Shapiro-Wilk Test"
    GROUP_COLUMN = "Group"
    SEM_COLUMN = "SEM"
    DIFF_FROM_MAX_COLUMN = 'Mean diff from max R'
    DIFF_FROM_MIN_COLUMN = 'Mean diff from min R'

    def __init__(self, csd_query, measurements_df):
        self.base = super(CsdQueryResult, self)
        self.base.__init__(csd_query.analysis_name)
        self.csd_query = csd_query
        self.measurements_df = measurements_df

        self.ref_codes = self.measurements_df.index.str.split('_', 1, True).get_level_values(0).unique().values
        self.structure_count = self.measurements_df.shape[0]
        self.means = self.measurements_df.mean()
        self.stds = self.measurements_df.std()
        self.shapiro_test_results = self.measurements_df.apply(self._shapiro_p_value, axis=0)
        self.bond_std_mean = self.stds.loc[self.csd_query._get_mapped_bond_names()].mean()
        self.angle_std_mean = self.stds.loc[self.csd_query._get_mapped_angle_names()].mean()

        self.summary_df = self._create_summary_data_frame()
        self.mean_df = self._create_means_data_frame()

    def recalculate_shapiro_test(self):
        self.shapiro_test_results = self.measurements_df.apply(self._shapiro_p_value, axis=0)

    def _create_summary_data_frame(self):
        summary_df = pd.DataFrame({self.NAME_COLUMN: [self.csd_query.name],
                                   self.MAX_R_FACTOR_COLUMN: [self.csd_query.conditions.max_r_factor],
                                   self.NO_DISORDER_COLUMN: [self.csd_query.conditions.no_disorder],
                                   self.STRUCTURE_COUNT_COLUMN: [self.structure_count],
                                   self.FILE_COUNT_COLUMN: [len(self.ref_codes)],
                                   self.REMOVE_OUTLIERS_COLUMN: [self.csd_query.conditions.remove_outliers],
                                   self.BOND_LENGTH_STD_MEAN_COLUMN: [self.bond_std_mean],
                                   self.BOND_ANGLE_STD_MEAN_COLUMN: [self.angle_std_mean],
                                   self.BOND_LENGTH_SEM_COLUMN: [self.bond_std_mean / np.sqrt(self.structure_count)],
                                   self.BOND_ANGLE_SEM_COLUMN: [self.angle_std_mean / np.sqrt(self.structure_count)],
                                   self.MAX_ESD_COLUMN: [self.csd_query.conditions.max_esd]})
        return summary_df

    def _create_means_data_frame(self):
        mean_df = self.means.to_frame(self.MEAN_COLUMN)
        mean_df[self.MEASUREMENT_COLUMN] = list(self.means.index)
        mean_df[self.STANDARD_DEVIATION_COLUMN] = self.stds
        mean_df[self.SEM_COLUMN] = self.stds / ((self.summary_df[self.STRUCTURE_COUNT_COLUMN] - 1) ** 0.5).values[0]
        mean_df[self.SHAPIRO_TEST_COLUMN] = self.shapiro_test_results
        mean_df.reset_index(drop=True)
        mean_df[self.NAME_COLUMN] = self.csd_query.name
        mean_df[self.MAX_R_FACTOR_COLUMN] = self.csd_query.conditions.max_r_factor
        mean_df[self.MAX_ESD_COLUMN] = self.csd_query.conditions.max_esd
        mean_df[self.NO_DISORDER_COLUMN] = self.csd_query.conditions.no_disorder
        mean_df[self.REMOVE_OUTLIERS_COLUMN] = self.csd_query.conditions.remove_outliers
        mean_df[self.GROUP_COLUMN] = mean_df.apply(lambda row: self._determine_measurement_group(row), axis=1)

        return mean_df

    def _determine_measurement_group(self, row):
        if row[self.MEASUREMENT_COLUMN] in self.csd_query._get_mapped_bond_names():
            return Measurement.bond
        if row[self.MEASUREMENT_COLUMN] in self.csd_query._get_mapped_angle_names():
            return Measurement.angle
        if row[self.MEASUREMENT_COLUMN] in self.csd_query._get_mapped_torsion_names():
            return Measurement.torsion
        else:
            return Measurement.other

    def save_restraints_to_file(self, measurement_order=None):
        restraints_file = self.base._create_results_save_path(self.csd_query.name + '_restraints.csv', subfolder="Restraints")
        logging.info("Saving restraint to: %s", restraints_file)

        restraints_df = pd.concat([self.means, self.stds], axis=1, ignore_index=False)
        restraints_df.columns = [self.MEAN_COLUMN, self.STANDARD_DEVIATION_COLUMN]
        restraints_df = restraints_df.reindex(measurement_order)
        restraints_df.to_csv(restraints_file, index=True)

    def save_ref_codes_to_file(self):
        ref_codes_file = self.base._create_results_save_path(self.csd_query.name + '_ref_codes.txt', subfolder="Ref codes")
        logging.info("Saving ref codes to: %s", ref_codes_file)

        with open(ref_codes_file, 'w') as f:
            f.write("\n".join(np.sort(self.ref_codes)))

    def save_subgroup_restraints(self, restraints_tuples, file_definitions, condition_mapper):
        for restraints_tuple in restraints_tuples:
            (df, groups, prefix) = restraints_tuple

            for group in groups.keys():
                logging.info(group)
                group_conditions = group.split("-")
                functional_argument = None

                condition_list = []
                for condition_column in group_conditions:
                    if condition_column == "All":
                        condition_list.append(["All"])
                    elif condition_column.startswith("Func"):
                        functional_argument = condition_column[5:-1]
                        logging.info(functional_argument)
                    else:
                        condition_list.append(np.unique(df.loc[:, condition_column]))

                for condition in itertools.product(*condition_list):
                    name = "__".join(condition)
                    group_df = df.copy()

                    if not group.startswith("All"):
                        for idx, col in enumerate(group_conditions):
                            if not col.startswith("Func"):
                                group_df = group_df.loc[(df.loc[:, col] == condition[idx]), :].copy()

                    if functional_argument is None:
                        restraints_df = self.create_subgroup_restraint_csv(group_df, groups[group], prefix + name)

                    for definition in file_definitions:
                        if self._is_matching_restraint(condition, definition.name) or self._is_universal_restraint(condition, file_definitions):
                            restraint_definition = self.create_restraintlib_restraint(restraints_df, condition, definition, group_conditions, condition_mapper, functional_argument)
                            definition.append_restraint(group, restraint_definition)

        for definition in file_definitions:
            self.create_retraintlib_file(definition)


    def _is_matching_restraint(self, conditions, definition_name):
        for c in conditions:
            if c.upper() in definition_name.upper().split(" "):
                return True
        return False

    def _is_universal_restraint(self, conditions, file_definitions):
        if conditions[0] == "All":
            return True

        for definition in file_definitions:
            if self._is_matching_restraint(conditions, definition.name):
                return False

        return True

    def log_counts(self):
        logging.info("----------------------------------")
        for sugar in self.measurements_df.Sugar.unique():
            for base in self.measurements_df.Base.unique():
                logging.info("%s-%s: %s", sugar, base,  self.measurements_df.loc[(self.measurements_df.Sugar == sugar) & (self.measurements_df.Base == base),:].shape[0])
        logging.info("----------------------------------")

    def create_retraintlib_file(self, definition):
        filename = self.base._create_results_save_path(definition.name.replace(" ", "_") + '.py', subfolder="Restraints")
        def_code_name = definition.name.replace(" ", "_").upper()

        angles = []
        conditions = []
        restraint_sets = []
        pdb_code_sets = []
        atom_name_sets = []
        atom_res_sets = []
        required_sets = []
        disallowed_sets = []
        distance_sets = []
        condition_distance_sets = []

        for r_set in definition.restraints:
            set_name = r_set.upper().replace("-", "_").replace("[", "_OF_").replace("]", "")

            for r_def in definition.restraints[r_set]:
                for r in r_def["restraints"]:
                    if not r[1] in angles and r[1][0] == "a":
                        angles.append(r[1])
                for c in r_def["conditions"]:
                    if c is not None and len(c) > 1 and not c[1] in conditions:
                        conditions.append(c[1])

            restraint_sets.append("{0}_{1}_RESTRAINTS = {2}".format(
                                  def_code_name,
                                  set_name,
                                  pprint.pformat(definition.restraints[r_set], indent=4, width=200)
                                      .replace("[   {", "[\n    {")
                                      .replace("}]", "}\n]")
                                      .replace("{", "{\n     ")
                                      .replace("}", "\n    }"))
            )
            pdb_code_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "PDB_CODES"))
            atom_name_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "ATOM_NAMES"))
            atom_res_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "ATOM_RES"))
            disallowed_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "DISALLOWED_CONDITION"))
            required_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "REQUIRED_CONDITION"))
            distance_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "DISTANCE_MEASURE"))
            condition_distance_sets.append("{0}_{1}_{2} = {0}_{2}".format(def_code_name, set_name, "CONDITION_DISTANCE_MEASURE"))

        disallowed = ""
        if definition.disallowed_atom_pairs is not None:
            disallowed = """
{0}_DISALLOWED_CONDITION = {1}
{2}
""".format(def_code_name,
           pprint.pformat(definition.disallowed_atom_pairs, indent=4).replace("]", "\n]").replace("[", "[\n "),
           "\n".join(disallowed_sets))

        with open(filename, 'w') as file:
            file.write("""{0}_PDB_CODES = {1}
{2}

{0}_ATOM_NAMES = {3}
{4}

{0}_ATOM_RES = {5}
{6}

{0}_REQUIRED_CONDITION = {7}
{8}
{14}
{0}_DISTANCE_MEASURE = {{
    'measure': 'euclidean_angles',
    'restraint_names': {9}
}}
{10}

{0}_CONDITION_DISTANCE_MEASURE = {{
    'measure': 'euclidean_angles',
    'restraint_names': {11}
}}
{12}

{13}""".format(def_code_name,
               definition.pdb_codes,
               "\n".join(pdb_code_sets),
               json.dumps(definition.atom_names, indent=4, separators=(',', ': '), sort_keys=True),
               "\n".join(atom_name_sets),
               json.dumps(definition.atom_res, indent=4, separators=(',', ': '), sort_keys=True),
               "\n".join(atom_res_sets),
               pprint.pformat(definition.required_atom_pairs, indent=4).replace("]", "\n]").replace("[", "[\n "),
               "\n".join(required_sets),
               angles,
               "\n".join(distance_sets),
               conditions,
               "\n".join(condition_distance_sets),
               "\n\n".join(restraint_sets),
               disallowed))

    def create_restraintlib_restraint(self, restraints_df, condition, file_definition, group_conditions,
                                      condition_mapper, functional_argument):
        restraint = dict()
        restraint["name"] = ""
        restraint["conditions"] = list()
        restraint["restraints"] = list()

        name = ""
        for i in range(len(condition)):
            condition_key = group_conditions[i] + "__" + condition[i]
            name = name + "__" + condition_key.replace("__", "=")
            if condition_key in condition_mapper and condition_mapper[condition_key] is not None:
                restraint["conditions"].append(self._disambiguate_restraint_condition(condition_mapper[condition_key], file_definition))

        restraint["name"] = file_definition.name.replace(" ", "_") + "==" + name[2:]

        for idx, r in restraints_df.iterrows():
            if self._all_atoms_in_restraint(idx, file_definition):
                restraint_data = [
                    self._get_restraint_type(idx),
                    self._get_restraint_name(idx, file_definition),
                    self._get_restraint_atoms(idx, file_definition)
                ]

                if functional_argument is None:
                    mean = round(r[self.MEAN_COLUMN], self._get_restraint_precision(idx))
                    sd = round(r[self.STANDARD_DEVIATION_COLUMN], self._get_restraint_precision(idx))

                    restraint_data.extend([mean, sd])
                else:
                    # TODO temp hack
                    if functional_argument == 'tau_max':
                        parameter_atoms = ["C1'", "C2'", "C3'", "C4'", "O4'"]
                    elif functional_argument == 'torsion_chi' and "N9" in file_definition.atom_names.values():
                        parameter_atoms = ["O4'", "C1'", "N9", "C4"]
                    elif functional_argument == 'torsion_chi' and "N1" in file_definition.atom_names.values():
                        parameter_atoms = ["O4'", "C1'", "N1", "C2"]

                    regressor_file = "-".join(condition) + "-" + idx.replace("/", " or ") + ".joblib"
                    regressor_parameter = [functional_argument, parameter_atoms]

                    restraint_data.extend([None, None, None, None, regressor_file, regressor_parameter])

                restraint["restraints"].append(restraint_data)

        return restraint

    def _all_atoms_in_restraint(self, name, file_definition):
        for restraint_atom in self._get_restraint_atoms(name, file_definition):
            if restraint_atom not in file_definition.atom_names.values():
                return False

        return True

    def _get_restraint_type(self, name):
        atom_count = len(name.split("/")[0].split("-"))

        if atom_count == 2:
            return "dist"
        elif atom_count == 3:
            return "angle"
        else:
            return "torsion"

    def _get_restraint_precision(self, name):
        type = self._get_restraint_type(name)

        if type == "dist":
            return 3
        elif type == "angle":
            return 1
        else:
            return 3

    def _get_restraint_name(self, name, file_definition):
        name = self._disambiguate_restraint_name(name, file_definition)
        prefix = self._get_restraint_type(name)[0]

        return prefix + name.replace("-", "")

    def _get_restraint_atoms(self, name, file_definition):
        name = self._disambiguate_restraint_name(name, file_definition)

        return name.split("-")

    def _disambiguate_restraint_condition(self, condition, file_definition):
        if condition is None:
            return None

        disambiguated_condition = list(condition)
        condition_name = condition[1]
        condition_atoms = condition[2]
        name_split = condition_name.split("/")

        if len(name_split) > 1:
            result = True

            for index in range(len(name_split)):
                for atom in condition_atoms:
                    atom_split =atom.split("/")
                    if len(atom_split) > 1 and atom_split[index] not in file_definition.atom_names.values():
                        result = False
                        break
                if result:
                    break

            disambiguated_condition_atoms = list(disambiguated_condition[2])
            for atom_idx in range(len(disambiguated_condition_atoms)):
                if len(disambiguated_condition_atoms[atom_idx].split("/")) > 1:
                    disambiguated_condition_atoms[atom_idx] = disambiguated_condition_atoms[atom_idx].split("/")[index]

            disambiguated_condition[1] = name_split[index]
            disambiguated_condition[2] = disambiguated_condition_atoms

        return disambiguated_condition

    def _disambiguate_restraint_name(self, name, file_definition):
        name_split = name.split("/")

        if len(name_split) == 1:
            return name
        else:
            result = True

            for index in range(len(name_split)):
                for atom in name_split[index].split("-"):
                    if atom not in file_definition.atom_names.values():
                        result = False
                        break
                if result:
                    break

            return name_split[index]

    def format_restraint_value(self, x):
        if len(x.name.split("-")) % 3 == 0:
            return '{0:.1f}({1:.0f})'.format(x[self.MEAN_COLUMN], x[self.STANDARD_DEVIATION_COLUMN] * 10)
        else:
            return '{0:.3f}({1:.0f})'.format(x[self.MEAN_COLUMN], x[self.STANDARD_DEVIATION_COLUMN] * 1000)

    def create_subgroup_restraint_csv(self, df, measurements, name):
        restraints_file = self.base._create_results_save_path(name + '_restraints.csv', subfolder="Restraints")
        logging.info("Saving restraint to: %s (%s)", restraints_file, measurements)

        means = df.loc[:, measurements].mean()
        stds = df.loc[:, measurements].std()
        count = df.shape[0]
        restraints_df = pd.concat([means, stds], axis=1, ignore_index=False)
        restraints_df.columns = [self.MEAN_COLUMN, self.STANDARD_DEVIATION_COLUMN]
        restraints_df.loc[:, "N"] = count
        restraints_df.loc[:, "Formatted"] = restraints_df.apply(self.format_restraint_value, axis=1)
        restraints_df.to_csv(restraints_file, index=True)

        return restraints_df

    def plot_histograms(self, measurement_order=None, width=10, height=9, dpi=PLOT_DPI, use_rice_rule=True,
                        show_temperature=False, color_column=None):
        gathered_df = self._get_gathered_plot_df()
        gathered_df[self.GROUP_COLUMN] = gathered_df.apply(lambda row: self._determine_measurement_group(row), axis=1)

        for group, group_enum in Measurement.__members__.items():
            plot_name = group_enum.name + ('' if color_column is None else '_' + str(color_column)) + '_' +\
                        self.csd_query.name + '_histograms.png'
            plot_df = gathered_df[(gathered_df[CsdQueryResult.GROUP_COLUMN].astype(str) == str(group_enum))]
            n = self.measurements_df.shape[0]

            if not plot_df.empty:
                order = self._compute_measurement_order(measurement_order)
                self._filter_and_order_measurements(plot_df, order, add_normality_test_asterisk=True)
                bins = self._estimate_number_of_bins(use_rice_rule)

                if plot_df.empty:
                    return

                p = (ggplot(plot_df, self._get_aes(None, color_column))
                     + geom_histogram(bins=bins)
                     + facet_wrap('~' + self.MEASUREMENT_COLUMN, scales='free_x')
                     + scale_x_continuous(breaks=mpl_breaks(nbins=4, steps=[1, 2, 4]))
                     + labs(x=self._get_measurement_plot_label(group_enum), y="Count")
                     + theme_bw()
                     + theme(panel_spacing_y=0.25, panel_spacing_x=0.2, axis_text_x=element_text(colour="black"),
                             axis_text_y=element_text(colour="black"), # legend_position=(0.84, 0.17)
                             )
                     + ggtitle(self.csd_query.name.replace("_", " ") + " (N=" + str(n) + ")"))

                if color_column is not None:
                    p = p + self._get_fill(color_column)

                if show_temperature:
                    p = p + geom_freqpoly(mapping=aes(color=CsdQuery.TEMP_COLUMN), bins=bins)

                self.base._save_plot(p, plot_name, width, height, dpi, subfolder=color_column)
                plt.clf()

    def plot_box(self, measurement_order=None, width=10, height=6, dpi=PLOT_DPI, use_rice_rule=True,
                        show_temperature=False, color_column=None, x_column=None, trim=True):
        gathered_df = self._get_gathered_plot_df()
        gathered_df[self.GROUP_COLUMN] = gathered_df.apply(lambda row: self._determine_measurement_group(row), axis=1)

        for group, group_enum in Measurement.__members__.items():
            plot_name = group_enum.name + ('' if color_column is None else '_' + str(color_column)) + '_' + \
                        ('' if x_column is None else '_' + str(x_column)) + '_' + \
                        self.csd_query.name + '_boxplot.png'
            plot_df = gathered_df[(gathered_df[CsdQueryResult.GROUP_COLUMN].astype(str) == str(group_enum))]
            n = self.measurements_df.shape[0]

            if not plot_df.empty:
                order = self._compute_measurement_order(measurement_order)
                self._filter_and_order_measurements(plot_df, order)
                bins = self._estimate_number_of_bins(use_rice_rule)

                if plot_df.empty:
                    return

                if x_column is None:
                    x_column = color_column

                if x_column is None and color_column is None:
                    p = (ggplot(plot_df, aes(self.MEASUREMENT_COLUMN, self.VALUE_COLUMN)) +
                         geom_violin(color="black", fill="#AAAAAA", trim=trim, width=0.33))
                else:
                    if x_column == color_column:
                        violin_width = min(1.0, plot_df.loc[:, color_column].unique().shape[0] / 3.0)
                    else:
                        violin_width = 1.0
                    p = (ggplot(plot_df, aes(x_column, self.VALUE_COLUMN, fill=color_column)) +
                         geom_violin(color="black", trim=trim, width=violin_width))

                p = (p + stat_summary(fun_data="mean_sdl", fun_args={"mult": 1}, geom="pointrange",
                                      position=position_dodge(width=0.9), color="black")
                     + facet_wrap('~' + self.MEASUREMENT_COLUMN, scales='free')
                     + labs(y=self._get_measurement_plot_label(group_enum))
                     + scale_y_continuous(breaks=mpl_breaks(nbins=5, steps=[1, 2]))
                     + theme_bw()
                     + theme(panel_spacing_x=0.5, panel_spacing_y=0.25, axis_text_x=element_text(colour="black", size=7),
                             axis_text_y=element_text(colour="black"), # legend_position=(0.84, 0.17)
                             )
                     + ggtitle(self.csd_query.name.replace("_", " ") + " (N=" + str(n) + ")"))

                if color_column is not None:
                    p = p + self._get_fill(color_column)
                    p = p + self._get_color(x_column)

                if show_temperature:
                    p = p + geom_freqpoly(mapping=aes(color=CsdQuery.TEMP_COLUMN), bins=bins)

                if group_enum is Measurement.angle:
                    height = height * 4.0 / 3.0

                self.base._save_plot(p, plot_name, width, height, dpi, subfolder=color_column)
                plt.clf()

    def plot_scatter(self, x, measurement_order=None, width=10, height=6, dpi=PLOT_DPI, color_column=None,
                     point_column=None, smooth_func=None, smooth_params=None, measurement_subset=None, suffix=None):
        column_mapper = {'Theta': u"Pseudorotation angle", 'T_max': u"Max. degree of pucker", "T3455": u"Gamma",
                         'TCHI': u"Chi", 'absChi': '|Chi|', 'absGamma': '|Gamma|', 'absChiDiff90': u'||Chi|-90°|',
                         'absGammaDiff90': u'||Gamma|-90°|'}

        gathered_df = self._get_normalized_gathered_plot_df(omit=[CsdQuery.TEMP_COLUMN, x])
        gathered_df[self.GROUP_COLUMN] = gathered_df.apply(lambda row: self._determine_measurement_group(row), axis=1)

        for group, group_enum in Measurement.__members__.items():
            plot_name = group_enum.name + '_' + x + '_' + \
                        ('' if color_column is None else '_' + str(color_column)) + '_' + \
                        ('' if point_column is None else '_' + str(point_column)) + '_' + \
                        ('' if suffix is None else '_' + str(suffix) + '_') + \
                        self.csd_query.name + '_scatter.png'
            plot_df = gathered_df[(gathered_df[CsdQueryResult.GROUP_COLUMN].astype(str) == str(group_enum))]
            n = self.measurements_df.shape[0]

            if not plot_df.empty:
                order = self._compute_measurement_order(measurement_order)
                self._filter_and_order_measurements(plot_df, order)

                if measurement_subset is not None:
                    plot_df = plot_df.loc[plot_df[self.MEASUREMENT_COLUMN].isin(measurement_subset), :]

                if plot_df.empty:
                    continue

                p = (ggplot(plot_df, self._get_aes(x, color_column, point_column))
                     + geom_point(alpha=0.5)
                     + facet_wrap('~' + self.MEASUREMENT_COLUMN, scales='free', ncol=4)
                     + scale_x_continuous(breaks=mpl_breaks(nbins=4, steps=[1, 2, 4]))
                     + scale_y_continuous(breaks=mpl_breaks(nbins=5, steps=[1, 2]))
                     + labs(x=column_mapper[x], y=self._get_measurement_plot_label(group_enum))
                     + theme_bw()
                     + theme(panel_spacing_x=0.5, panel_spacing_y=0.25, axis_text_x=element_text(colour="black"),
                             axis_text_y=element_text(colour="black"), # legend_position=(0.84, 0.17)
                             )
                     + ggtitle(self.csd_query.name.replace("_", " ") + " (N=" + str(n) + ")"))

                if smooth_func is not None:
                    p = (p + geom_smooth(alpha=0.25, method=smooth_func, fullrange=True, method_args=smooth_params, n=360)
                         + scale_linetype_manual(values=["solid", "dashed", "dotted"]))

                if color_column is not None:
                    p = p + self._get_fill(color_column)
                    p = p + self._get_color(color_column)

                self.base._save_plot(p, plot_name, width, height, dpi, subfolder=color_column)
                plt.clf()

    def _plot_custom_regressions(self, measurements_order, gpr_smoother, bayesian_ridge_smoother):
        self.plot_scatter("T_max", measurements_order, color_column="Sugar", point_column="Conformation",
                                     width=10, height=1.61, suffix="br",
                                     measurement_subset=["C1'-C2'-C3'", "C2'-C3'-C4'", "C3'-C4'-O4'", "C1'-O4'-C4'"],
                                     smooth_func=bayesian_ridge_smoother,
                                     smooth_params={
                                         "normalize": True,
                                     })
        self.plot_scatter("TCHI", measurements_order,
                                     width=4.5, height=1.61, suffix="gpr",
                                     measurement_subset=["C1'-N1/C1'-N9", "C1'-O4'"],
                                     smooth_func=gpr_smoother,
                                     smooth_params={
                                         # parameters set to achieve faster convergence - actual regressors and exact plots are in the Regressors folder
                                         "kernel": (ExpSineSquared(periodicity_bounds=(180, 180)) + WhiteKernel(noise_level_bounds=(1e-6, 1e7))),
                                         "n_restarts_optimizer": 1,
                                         "random_state": 23,
                                         "normalize_y": True,
                                     })
        self.plot_scatter("TCHI", measurements_order, color_column="Base",
                                     width=10, height=1.61, suffix="gpr",
                                     measurement_subset=["N1-C1'-C2'/N9-C1'-C2'", "C1'-N1-C2/C1'-N9-C4", "C1'-N1-C6/C1'-N9-C8", "N1-C1'-O4'/N9-C1'-O4'"],
                                     smooth_func=gpr_smoother,
                                     smooth_params={
                                         # parameters set to achieve faster convergence - actual regressors and exact plots are in the Regressors folder
                                         "kernel": (ExpSineSquared(periodicity_bounds=(360, 360)) + WhiteKernel(noise_level_bounds=(1, 1e7))),
                                         "n_restarts_optimizer": 1,
                                         "random_state": 23,
                                         "normalize_y": True,
                                     })

    def _get_aes(self, x, color_column, point_column=None):
        if x is None:
            if color_column is not None:
                aesthetic = aes(self.VALUE_COLUMN, fill=color_column)
            else:
                aesthetic = aes(self.VALUE_COLUMN)
        else:
            if color_column is not None:
                if point_column is not None:
                    aesthetic = aes(x, self.VALUE_COLUMN, fill=color_column, color=color_column,
                                    shape=point_column, linetype=point_column)
                else:
                    aesthetic = aes(x, self.VALUE_COLUMN, fill=color_column, color=color_column)
            else:
                aesthetic = aes(x, self.VALUE_COLUMN)
        return aesthetic

    def _get_pallette(self, color_column):
        if color_column == "Conformation":
            return ['#77AADD', '#EE8866', '#EEDD88']
        elif color_column == "Gamma":
            return ['#CC3311', '#0077BB', '#009988', '#EE7733', '#EE3377', '#BBBBBB', "#33BBEE", "#AA4499"]
        elif color_column == "Chi":
            return ["#117733", "#882255", '#AA4499', '#44AA99']
        elif color_column == "Chi_syn":
            return ['#CC3311', '#0077BB']
        elif color_column == "Gamma_syn":
            return ['#CC3311', '#0077BB']
        elif color_column == "Sugar":
            return ['#EE3377', '#33BBEE']
        elif color_column == "Base":
            return ["#EE7733", "#009988"]
        else:
            return ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', "#BBCC33", "#DDDDDD"]

    def _get_fill(self, color_column):
        palette = self._get_pallette(color_column)
        return scale_fill_manual(palette, limits=self.measurements_df[color_column].dtype.categories)

    def _get_color(self, color_column):
        palette = self._get_pallette(color_column)
        return scale_color_manual(palette, limits=self.measurements_df[color_column].dtype.categories)

    def _get_measurement_plot_label(self, group_enum):
        if group_enum == Measurement.bond:
            x_lab = u"Bond length [Å]"
        elif group_enum == Measurement.angle:
            x_lab = u"Bond angle [°]"
        else:
            x_lab = u"Torsion angle [°]"
        return x_lab

    def plot_2d_histogram(self, measurement_order=None, width=10, height=9, dpi=PLOT_DPI):
        gathered_df = self._get_normalized_gathered_plot_df(omit=[CsdQuery.TEMP_COLUMN, 'Theta', 'T_max', 'TCHI', 'T3455'])
        gathered_df[self.GROUP_COLUMN] = gathered_df.apply(lambda row: self._determine_measurement_group(row), axis=1)

        for group, group_enum in Measurement.__members__.items():
            base_plot_name = self.csd_query.name + '_' + group_enum.name
            plot_df = gathered_df[(gathered_df[CsdQueryResult.GROUP_COLUMN].astype(str) == str(group_enum))]

            if not plot_df.empty:
                order = self._compute_measurement_order(measurement_order)
                self._filter_and_order_measurements(plot_df, order)

                if plot_df.empty:
                    return

                plots = [(('Theta',u"Pseudorotation angle"), ('T_max', u"Max. degree of pucker")),
                         (('Theta', u"Pseudorotation angle"), ("T3455", u"Gamma")),
                         (('Theta', u"Pseudorotation angle"), ('TCHI', u"Chi")),
                         (('TCHI', u"Chi"), ("T3455", u"Gamma")),
                         (('TCHI', u"Chi"), ('T_max', u"Max. degree of pucker")),
                         (("T3455", u"Gamma"), ('T_max', u"Max. degree of pucker")),
                         ]

                for kvps in plots:
                    plot_name = base_plot_name + kvps[0][0] + kvps[1][0] + '_2dbin.png'
                    p = (ggplot(plot_df, aes(x=kvps[0][0], y=kvps[1][0], color=self.VALUE_COLUMN))
                         + geom_point(alpha=0.7)
                         + scale_color_distiller(type='div', palette='Spectral')
                         + facet_wrap('~' + self.MEASUREMENT_COLUMN)
                         + labs(x=kvps[0][1], y=kvps[1][1])
                         + theme_bw()
                         + theme(panel_spacing_y=0.25, axis_text_x=element_text(colour="black"),
                                 axis_text_y=element_text(colour="black")))

                    self.base._save_plot(p, plot_name, width, height, dpi)
                    plt.clf()

    def _get_normalized_gathered_plot_df(self, omit=[CsdQuery.TEMP_COLUMN]):
        plot_df = self.measurements_df.copy()
        plot_df = plot_df.drop(omit, axis=1)
        # plot_df = self._log_scale_df_columns(plot_df)

        for column in omit:
            plot_df.loc[plot_df.index, column] = self.measurements_df.loc[plot_df.index, column].values

        gather_columns = self.csd_query.measurement_name_mapping.values()
        gather_columns = [c for c in gather_columns if c not in omit]
        plot_df = self._gather(plot_df, self.MEASUREMENT_COLUMN, self.VALUE_COLUMN, gather_columns)

        return plot_df

    def _log_scale_df_columns(self, df):
        df = df - df.min() + df.std()

        return np.log(df / df.min()) / np.log(df.max() / df.min())

    def _lin_scale_df_columns(self, df):
        return (df / df.min()) / (df.max() - df.min())

    def _standardize_df_columns(self, df):
        return (df - df.mean()) / df.std()

    def _get_gathered_plot_df(self):
        plot_df = self.measurements_df.copy()
        plot_df.loc[:, CsdQuery.TEMP_COLUMN] = pd.cut(plot_df[CsdQuery.TEMP_COLUMN], bins=[0, 149, 1000],
                                                      include_lowest=True, labels=[u'< 150K', u'≥ 150K'])
        gather_columns = self.csd_query.measurement_name_mapping.values()
        gather_columns.remove(CsdQuery.TEMP_COLUMN)
        plot_df = self._gather(plot_df, self.MEASUREMENT_COLUMN, self.VALUE_COLUMN, gather_columns)

        return plot_df

    def _compute_measurement_order(self, measurement_order):
        if measurement_order is not None:
            order = list(measurement_order)  # copy required
            order.append(CsdQuery.TEMP_COLUMN)
        else:
            order = list(self.measurements_df.columns.values)
        return order

    def _filter_and_order_measurements(self, plot_df, order, add_normality_test_asterisk=False):
        for measurement in self.shapiro_test_results.index:
            if add_normality_test_asterisk and self.shapiro_test_results[measurement] < 0.05:
                ids = plot_df[plot_df[self.MEASUREMENT_COLUMN] == measurement].index
                plot_df.loc[ids, self.MEASUREMENT_COLUMN] = measurement + '*'
                if measurement in order:
                    order[order.index(measurement)] = measurement + '*'
        plot_df[self.MEASUREMENT_COLUMN] = plot_df[self.MEASUREMENT_COLUMN].astype(
            pd.api.types.CategoricalDtype(ordered=True, categories=order))
        plot_df.dropna(subset=[self.MEASUREMENT_COLUMN], inplace=True)

    def _estimate_number_of_bins(self, use_rice_rule):
        if use_rice_rule:
            return math.ceil(2 * self.measurements_df.shape[0] ** (1.0 / 3.0))
        else:
            return math.ceil(self.measurements_df.shape[0] ** (1.0 / 2.0))

    @staticmethod
    def _gather(df, key, value, cols):
        id_vars = [col for col in df.columns if col not in cols]
        id_values = cols
        var_name = key
        value_name = value
        return pd.melt(df, id_vars, id_values, var_name, value_name)

    @staticmethod
    def _shapiro_p_value(x):
        try:
            return scipy.stats.shapiro(x)[1]
        except:
            return 1;


class CsdMultiQueryAnalysis(CsdAnalysisComponent):
    def __init__(self, analysis_name, query_folder, r_factors, non_disordered, esds, remove_outliers, name_mappings):
        self.base = super(CsdMultiQueryAnalysis, self)
        self.base.__init__(analysis_name)
        self.query_folder = query_folder
        self.r_factors = r_factors
        self.non_disordered = non_disordered
        self.esds = esds
        self.remove_outliers = remove_outliers
        self.name_mappings = name_mappings

    def run_analysis(self):
        logging.info("Performing multiple CSD queries using folder: %s", self.query_folder)

        NAME = 0;
        R = 1;
        NO_DISORDER = 2;
        ESD = 3;
        OUTLIERS = 4;
        query_names = CsdQuery.get_query_names_from_folder(self.query_folder)
        analysis_result = CsdMultiQueryAnalysisResults(self.analysis_name)

        for condition in itertools.product(query_names, self.r_factors, self.non_disordered,
                                           self.esds, self.remove_outliers):
            con_query_path = os.path.join(self.query_folder, condition[NAME] + ".con")
            csd_query_conditions = CsdQueryConditions(condition[R], condition[NO_DISORDER], condition[ESD],
                                                      condition[OUTLIERS], self.name_mappings[condition[NAME]].keys())
            csd_query = CsdQuery(condition[NAME], con_query_path, self.name_mappings[condition[NAME]],
                                 csd_query_conditions, analysis_name=self.analysis_name)
            csd_query_result = csd_query.run()
            analysis_result.append_single_query_result(csd_query_result)

        analysis_result._calculate_mean_diff()
        return analysis_result


class CsdMultiQueryAnalysisResults(CsdAnalysisComponent):
    PLOT_STRUCTURES_COLUMN = 'Structures'
    PLOT_OUTLIERS_COLUMN = 'Outliers'
    PLOT_ESD_COLUMN = 'ESD'

    def __init__(self, analysis_name):
        self.base = super(CsdMultiQueryAnalysisResults, self)
        self.base.__init__(analysis_name)
        self.std_analysis_df = None
        self.mean_analysis_df = None
        self._is_initialized = False

    @staticmethod
    def read_from_csv_files(analysis_name):
        instance = CsdMultiQueryAnalysisResults(analysis_name)
        instance.std_analysis_df = \
            pd.read_csv(instance.base._create_results_save_path(analysis_name + "_std_analysis.csv"))
        instance.mean_analysis_df = \
            pd.read_csv(instance.base._create_results_save_path(analysis_name + "_mean_analysis.csv"))

        return instance

    def append_single_query_result(self, csd_query_result):
        if not self._is_initialized:
            self.std_analysis_df = csd_query_result.summary_df
            self.mean_analysis_df = csd_query_result.mean_df
            self._is_initialized = True
        else:
            self.std_analysis_df = self.std_analysis_df.append(csd_query_result.summary_df, ignore_index=True)
            self.mean_analysis_df = self.mean_analysis_df.append(csd_query_result.mean_df, ignore_index=True)

    def save_to_csvs(self):
        std_df_file_name = self.analysis_name + "_std_analysis.csv"
        mean_df_file_name = self.analysis_name + "_mean_analysis.csv"

        self.std_analysis_df.to_csv(self.base._create_results_save_path(std_df_file_name), index=False)
        self.mean_analysis_df.to_csv(self.base._create_results_save_path(mean_df_file_name), index=False)

    def plot_mean_analysis(self, width=6, height=9, dpi=PLOT_DPI):
        plot_df = self.std_analysis_df.copy()
        self._map_column_names_for_ggplot(plot_df)

        measurements = {
            CsdQueryResult.STRUCTURE_COUNT_COLUMN: "Structure count",
            CsdQueryResult.BOND_LENGTH_SEM_COLUMN: u'Avg. bond length SEM [Å]',
            CsdQueryResult.BOND_ANGLE_SEM_COLUMN: u'Avg. bond angle SEM [°]',
            CsdQueryResult.BOND_LENGTH_STD_MEAN_COLUMN: u'Avg. bond length standard deviation [Å]',
            CsdQueryResult.BOND_ANGLE_STD_MEAN_COLUMN:  u'Avg. bond angle standard deviation [°]'
        }

        for measurement in measurements.keys():
            plot_name = self.analysis_name + '_' + measurement + '_mean_analysis.png'

            p = (ggplot(plot_df, aes(CsdQueryResult.MAX_R_FACTOR_COLUMN, measurement, color=self.PLOT_STRUCTURES_COLUMN,
                                     linetype=self.PLOT_OUTLIERS_COLUMN, shape=self.PLOT_OUTLIERS_COLUMN))
                 + geom_point()
                 + geom_line()
                 + facet_grid(CsdQueryResult.NAME_COLUMN + '~' + self.PLOT_ESD_COLUMN)
                 + theme_bw()
                 + labs(x="Max R-factor [%]", y=measurements[measurement])
                 + scale_color_manual(['#EE7733', '#0077BB'])
                 + theme(legend_position="top", legend_box="horizontal", legend_title=element_blank(),
                         axis_text_x=element_text(colour="black"), axis_text_y=element_text(colour="black")))

            self.base._save_plot(p, plot_name, width, height, dpi)
            plt.clf()

    def plot_measurement_differences(self, remove_outliers, no_disorder, max_esd, width=6, height=9):
        max_r = self.mean_analysis_df[CsdQueryResult.MAX_R_FACTOR_COLUMN].max()
        min_r = self.mean_analysis_df[CsdQueryResult.MAX_R_FACTOR_COLUMN].min()

        for group, group_enum in Measurement.__members__.items():
            plot_labels = []

            plot_df = self.mean_analysis_df[(self.mean_analysis_df[CsdQueryResult.GROUP_COLUMN].astype(str) == str(group_enum)) &
                                            (self.mean_analysis_df[CsdQueryResult.REMOVE_OUTLIERS_COLUMN] == remove_outliers) &
                                            (self.mean_analysis_df[CsdQueryResult.NO_DISORDER_COLUMN] == no_disorder) &
                                            (self.mean_analysis_df[CsdQueryResult.MAX_ESD_COLUMN] == max_esd) &
                                            (self.mean_analysis_df[CsdQueryResult.MEASUREMENT_COLUMN] != CsdQuery.TEMP_COLUMN)]
            if not plot_df.empty:
                plot_labels.append(self._create_plot_diff_column(plot_df, group_enum, max_r,
                                                                 CsdQueryResult.DIFF_FROM_MAX_COLUMN))
                plot_labels.append(self._create_plot_diff_column(plot_df, group_enum, min_r,
                                                                 CsdQueryResult.DIFF_FROM_MIN_COLUMN))
                for stat in plot_labels:
                    self._plot_diff(plot_df, stat, width, height)

    def _plot_diff(self, plot_df, stat, width, height, dpi=PLOT_DPI):
        plot_name = self.analysis_name + '_' + stat + '.png'

        p = (ggplot(plot_df, aes(CsdQueryResult.MAX_R_FACTOR_COLUMN, stat, color=CsdQueryResult.MEASUREMENT_COLUMN))
             + geom_point()
             + geom_line()
             + facet_grid(CsdQueryResult.NAME_COLUMN + '~.')
             + labs(x='Max R-factor [%]')
             + theme_bw()
             + theme(legend_position="top", legend_box="horizontal", legend_title=element_blank(),
                     axis_text_x=element_text(colour="black"), axis_text_y=element_text(colour="black")))

        self.base._save_plot(p, plot_name, width, height, dpi)

    @staticmethod
    def _create_plot_diff_column(plot_df, group_enum, diff_r, diff_column):
        if group_enum is Measurement.bond:
            lab = u'Difference in bond length means compared to R=' + str(diff_r) + u'% [Å]'
            plot_df.loc[:, lab] = plot_df[diff_column]
        elif group_enum is Measurement.angle:
            lab = u'Difference in bond angle means compared to R=' + str(diff_r) + u'% [°]'
            plot_df.loc[:, lab] = plot_df[diff_column]
        elif group_enum is Measurement.torsion:
            lab = u'Difference in torsion angle means compared to R=' + str(diff_r) + u'% [°]'
            plot_df.loc[:, lab] = plot_df[diff_column]
        else:
            lab = u'Difference in other measurements compared to R=' + str(diff_r) + u'% [°]'
            plot_df.loc[:, lab] = plot_df[diff_column]

        return lab

    @classmethod
    def _map_column_names_for_ggplot(cls, df):
        df[cls.PLOT_STRUCTURES_COLUMN] = df.loc[:, CsdQueryResult.NO_DISORDER_COLUMN].map({
            True: 'Only non-disordered structures',
            False: 'All structures'})
        df[cls.PLOT_OUTLIERS_COLUMN] = df.loc[:, CsdQueryResult.REMOVE_OUTLIERS_COLUMN].map(
            {True: 'Outlier removal',
             False: 'No outlier removal',
             Measurement.bond: 'Outlier removal (bond analysis)',
             Measurement.angle: 'Outlier removal (angle analysis)',
             Measurement.torsion: 'Outlier removal (torsion analysis)'})
        df[cls.PLOT_ESD_COLUMN] = df.loc[:, 'Max e.s.d. (C-C)'].map(
            {0.005: u'Average $\sigma$(C-C) < 0.005 Å',
             0.01: u'Average $\sigma$(C-C) < 0.01 Å',
             0.03: u'Average $\sigma$(C-C) < 0.03 Å',
             1: u'Average $\sigma$(C-C) unbounded'})

    def _calculate_mean_diff(self):
        self.mean_analysis_df[CsdQueryResult.DIFF_FROM_MAX_COLUMN] = 0.0
        self.mean_analysis_df[CsdQueryResult.DIFF_FROM_MIN_COLUMN] = 0.0

        for index, row in self.mean_analysis_df.iterrows():
            self.mean_analysis_df.at[index, CsdQueryResult.DIFF_FROM_MAX_COLUMN] = \
                self._get_max_for_row(row)[CsdQueryResult.MEAN_COLUMN] - row[CsdQueryResult.MEAN_COLUMN]
            self.mean_analysis_df.at[index, CsdQueryResult.DIFF_FROM_MIN_COLUMN] = \
                self._get_min_for_row(row)[CsdQueryResult.MEAN_COLUMN] - row[CsdQueryResult.MEAN_COLUMN]

    def _get_min_for_row(self, row):
        minimum_max_r = self.mean_analysis_df[CsdQueryResult.MAX_R_FACTOR_COLUMN].min()
        return self._get_matching_row_with_r(row, minimum_max_r)

    def _get_max_for_row(self, row):
        maximum_max_r = self.mean_analysis_df[CsdQueryResult.MAX_R_FACTOR_COLUMN].max()
        return self._get_matching_row_with_r(row, maximum_max_r)

    def _get_matching_row_with_r(self, row, max_r):
        matching_row = self.mean_analysis_df[
            (self.mean_analysis_df[CsdQueryResult.MAX_R_FACTOR_COLUMN] == max_r) &
            (self.mean_analysis_df[CsdQueryResult.NAME_COLUMN] == row[CsdQueryResult.NAME_COLUMN]) &
            (self.mean_analysis_df[CsdQueryResult.MEASUREMENT_COLUMN] == row[CsdQueryResult.MEASUREMENT_COLUMN]) &
            (self.mean_analysis_df[CsdQueryResult.MAX_ESD_COLUMN] == row[CsdQueryResult.MAX_ESD_COLUMN]) &
            (self.mean_analysis_df[CsdQueryResult.REMOVE_OUTLIERS_COLUMN] == row[CsdQueryResult.REMOVE_OUTLIERS_COLUMN]) &
            (self.mean_analysis_df[CsdQueryResult.NO_DISORDER_COLUMN] == row[CsdQueryResult.NO_DISORDER_COLUMN])]

        return matching_row


def _plot_rmsd_analysis(df, filename, width=2.5, height=9, dpi=PLOT_DPI):
    plot_df = df.loc[df['Group'] == 'RMSD bonds', :]
    p = (ggplot(plot_df, aes('Baseline', 'RMSD', fill='Library'))
         + geom_bar(stat='identity', position='dodge')
         + facet_grid('Base~Group')
         + labs(y=u'RMSD [Å]')
         + theme_bw()
         + scale_fill_manual(['#CC3311', '#009988', '#BBBBBB'])
         + theme(legend_position="top", legend_box="horizontal", legend_title=element_blank(),
                 axis_text_x=element_text(colour="black"), axis_text_y=element_text(colour="black"))
    )

    ggsave(plot=p, filename=filename + '_bonds.png', dpi=dpi, width=width, height=height)

    plot_df = df.loc[df['Group'] == 'RMSD angles', :]
    p = (ggplot(plot_df, aes('Baseline', 'RMSD', fill='Library'))
         + geom_bar(stat='identity', position='dodge')
         + facet_grid('Base~Group')
         + labs(y=u'RMSD [°]')
         + theme_bw()
         + scale_fill_manual(['#CC3311', '#009988', '#BBBBBB'])
         + theme(legend_position="top", legend_box="horizontal", legend_title=element_blank(),
                 axis_text_x=element_text(colour="black"), axis_text_y=element_text(colour="black")))

    ggsave(plot=p, filename=filename + '_angles.png', dpi=dpi, width=width, height=height)
    plt.clf()


def plot_histogram_comparison(combined_measurements_df, width=10, height=10, dpi=PLOT_DPI,
                              omit=[CsdQuery.TEMP_COLUMN, 'Theta', 'T_max', 'TCHI', 'T3455', 'Family',
                                    "P_group", "Chi_sign", "Gamma_sign", "Gamma_group", "Base", "Query", "Sugar"]):
    for measurement in combined_measurements_df:
        if measurement in omit:
            continue

        plot_df = combined_measurements_df[[measurement, "P_group", "Chi_sign", "Gamma_sign", "Gamma_group",
                                            "Sugar", "Base"]]
        plot_df = plot_df.dropna()
        plot_name = measurement + '_histogram.png'

        # p = (ggplot(plot_df, aes(x=measurement))
        #      + geom_histogram()
        #      + facet_grid('Gamma_sign~Sugar+Base')
        #      + theme_bw()
        #      + theme(panel_spacing_y=0.25, axis_text_x=element_text(colour="black"),
        #              axis_text_y=element_text(colour="black")))
        #
        # ggsave(plot=p, filename=plot_name, dpi=dpi, width=width, height=height)
        # plt.clf()
        #
        # plot_name = measurement + '_histogram_simplified.png'

        p = (ggplot(plot_df, aes(x=measurement, fill="P_group"))
             + geom_histogram()
             # + geom_vline(aes(xintercept="mean(" + measurement + ")"))
             # + geom_text(aes(label="mean="))
             + facet_grid('.~Sugar+Base')
             + theme_bw()
             + theme(panel_spacing_y=0.25, axis_text_x=element_text(colour="black"),
                     axis_text_y=element_text(colour="black")))

        ggsave(plot=p, filename=plot_name, dpi=dpi, width=width, height=2.5)
        plt.clf()