# coding: utf-8
# Authors: Marcin Kowiel <mkowiel@ump.edu.pl>
from __future__ import print_function

from collections import defaultdict
from glob import glob
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

from ccdc.protein import Protein

from download import download_from_query_file, download_from_csv_file

pd.set_option('display.max_columns', None)

DEBUG = False
DNA_RESIDUES = ['DA', 'DC', 'DG', 'DT', 'DU']
RNA_RESIDUES = ['A', 'C', 'G', 'T', 'U']
KEEP_RESIDUES = DNA_RESIDUES + RNA_RESIDUES
KEEP_DISORDERED = False

RIBOSE_PYRIMIDINE_ATOMS = ["C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N1", "C2", "C6", ]
RIBOSE_PURINE_ATOMS = ["C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", "N9", "C4", "C8", ]
DEOXYRIBOSE_PYRIMIDINE_ATOMS = ["C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",  "N1", "C2", "C6",]
DEOXYRIBOSE_PURINE_ATOMS = ["C5'", "O5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'", "N9", "C4", "C8", ]

RIBOSE_PYRIMIDINE_SUGAR_BONDS = [
    ("C5'", "O5'"), ("C4'", "C5'"), ("C4'", "O4'"), ("C1'", "O4'"), ("C3'", "C4'"), ("C3'", "O3'"), ("C2'", "C3'"),
    ("C1'", "C2'"), ("C2'", "O2'"), ("C1'", "N1"),
]
RIBOSE_PURINE_SUGAR_BONDS = [
    ("C5'", "O5'"), ("C4'", "C5'"), ("C4'", "O4'"), ("C1'", "O4'"), ("C3'", "C4'"), ("C3'", "O3'"), ("C2'", "C3'"), 
    ("C1'", "C2'"), ("C2'", "O2'"), ("C1'", "N9"),
]
DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS = [
    ("C5'", "O5'"), ("C4'", "C5'"), ("C4'", "O4'"), ("C1'", "O4'"), ("C3'", "C4'"), ("C3'", "O3'"), ("C2'", "C3'"), 
    ("C1'", "C2'"), ("C1'", "N1"),
]
DEOXYRIBOSE_PURINE_SUGAR_BONDS = [
    ("C5'", "O5'"), ("C4'", "C5'"), ("C4'", "O4'"), ("C1'", "O4'"), ("C3'", "C4'"), ("C3'", "O3'"), ("C2'", "C3'"), 
    ("C1'", "C2'"), ("C1'", "N9"),
]
RIBOSE_PYRIMIDINE_SUGAR_ANGLES = [
    ("C4'", "C5'", "O5'"), ("C5'", "C4'", "O4'"), ("C3'", "C4'", "C5'"), ("C2'", "C1'", "O4'"), ("C1'", "C2'", "C3'"),
    ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C1'", "O4'", "C4'"), ("C2'", "C3'", "O3'"), ("C4'", "C3'", "O3'"),
    ("C3'", "C2'", "O2'"), ("C1'", "C2'", "O2'"), ("N1", "C1'", "C2'"), ("N1", "C1'", "O4'"), ("C1'", "N1", "C2"),
    ("C1'", "N1", "C6"),
]
RIBOSE_PURINE_SUGAR_ANGLES = [
    ("C4'", "C5'", "O5'"), ("C5'", "C4'", "O4'"), ("C3'", "C4'", "C5'"), ("C2'", "C1'", "O4'"), ("C1'", "C2'", "C3'"),
    ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C1'", "O4'", "C4'"), ("C2'", "C3'", "O3'"), ("C4'", "C3'", "O3'"),
    ("C3'", "C2'", "O2'"), ("C1'", "C2'", "O2'"), ("N9", "C1'", "C2'"), ("N9", "C1'", "O4'"), ("C1'", "N9", "C4"),
    ("C1'", "N9", "C8"),
]
DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES = [
    ("C4'", "C5'", "O5'"), ("C5'", "C4'", "O4'"), ("C3'", "C4'", "C5'"), ("C2'", "C1'", "O4'"), ("C1'", "C2'", "C3'"),
    ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C1'", "O4'", "C4'"), ("C2'", "C3'", "O3'"), ("C4'", "C3'", "O3'"),
    ("N1", "C1'", "C2'"), ("N1", "C1'", "O4'"), ("C1'", "N1", "C2"), ("C1'", "N1", "C6"),
]
DEOXYRIBOSE_PURINE_SUGAR_ANGLES = [
    ("C4'", "C5'", "O5'"), ("C5'", "C4'", "O4'"), ("C3'", "C4'", "C5'"), ("C2'", "C1'", "O4'"), ("C1'", "C2'", "C3'"),
    ("C2'", "C3'", "C4'"), ("C3'", "C4'", "O4'"), ("C1'", "O4'", "C4'"), ("C2'", "C3'", "O3'"), ("C4'", "C3'", "O3'"),
    ("N9", "C1'", "C2'"), ("N9", "C1'", "O4'"), ("C1'", "N9", "C4"), ("C1'", "N9", "C8"),
]

SUGARS_BONDS_MAP = {
    'DA': DEOXYRIBOSE_PURINE_SUGAR_BONDS,
    'DC': DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS,
    'DG': DEOXYRIBOSE_PURINE_SUGAR_BONDS,
    'DT': DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS,
    'DU': DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS,
    'A': RIBOSE_PURINE_SUGAR_BONDS,
    'C': RIBOSE_PYRIMIDINE_SUGAR_BONDS,
    'G': RIBOSE_PURINE_SUGAR_BONDS,
    'T': RIBOSE_PYRIMIDINE_SUGAR_BONDS,
    'U': RIBOSE_PYRIMIDINE_SUGAR_BONDS,
}

SUGARS_ANGLES_MAP = {
    'DA': DEOXYRIBOSE_PURINE_SUGAR_ANGLES,
    'DC': DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES,
    'DG': DEOXYRIBOSE_PURINE_SUGAR_ANGLES,
    'DT': DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES,
    'DU': DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES,
    'A': RIBOSE_PURINE_SUGAR_ANGLES,
    'C': RIBOSE_PYRIMIDINE_SUGAR_ANGLES,
    'G': RIBOSE_PURINE_SUGAR_ANGLES,
    'T': RIBOSE_PYRIMIDINE_SUGAR_ANGLES,
    'U': RIBOSE_PYRIMIDINE_SUGAR_ANGLES,
}

PO4_BONDS = [
    ("OP1", "P"), ("OP2", "P"), ("O5'", "P"), ("O3'", "P", 0, +1), ("O3'", "C3'"), ("O5'", "C5'"),
    ("OP1", "P"), ("OP2", "P"), ("O5'", "P"), ("O3'", "P", 0, +1), ("O3'", "C3'"), ("O3'", "C3'")
]
PO4_ANGLES = [
    ("OP1", "P", "OP2"),  ("OP1", "P", "O5'"), ("OP2", "P", "O5'"), ("P", "O5'", "C5'"),
    ("O3'", "P", "O5'", 0, +1, +1), ("P", "O3'", "C3'", +1, 0, 0), ("OP2", "P", "O3'", +1, +1, 0),
    ("OP1", "P", "O3'", +1, +1, 0),
]

PO4_BONDS_MAP = {
    'DA': PO4_BONDS,
    'DC': PO4_BONDS,
    'DG': PO4_BONDS,
    'DT': PO4_BONDS,
    'DU': PO4_BONDS,
    'A': PO4_BONDS,
    'C': PO4_BONDS,
    'G': PO4_BONDS,
    'T': PO4_BONDS,
    'U': PO4_BONDS,
}

PO4_ANGLES_MAP = {
    'DA': PO4_ANGLES,
    'DC': PO4_ANGLES,
    'DG': PO4_ANGLES,
    'DT': PO4_ANGLES,
    'DU': PO4_ANGLES,
    'A': PO4_ANGLES,
    'C': PO4_ANGLES,
    'G': PO4_ANGLES,
    'T': PO4_ANGLES,
    'U': PO4_ANGLES,
}

ADENINE_BONDS = [('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C5', 'N7'),
                 ('N7', 'C8'), ('C8', 'N9'), ('N9', 'C4'), ('C6', 'N6')]
ADENINE_ANGLES = [('C6', 'N1', 'C2'), ('N1', 'C2', 'N3'), ('C2', 'N3', 'C4'), ('N3', 'C4', 'C5'), ('C4', 'C5', 'C6'),
                  ('C5', 'C6', 'N1'), ('N3', 'C4', 'N9'), ('C6', 'C5', 'N7'), ('C5', 'C4', 'N9'), ('C4', 'N9', 'C8'),
                  ('N9', 'C8', 'N7'), ('C8', 'N7', 'C5'), ('N7', 'C5', 'C4'), ('N6', 'C6', 'N1'), ('N6', 'C6', 'C5')]
GUANINE_BONDS = [('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C5', 'N7'),
                 ('N7', 'C8'), ('C8', 'N9'), ('N9', 'C4'), ('C6', 'O6'), ('C2', 'N2')]
GUANINE_ANGLES = [('C6', 'N1', 'C2'), ('N1', 'C2', 'N3'), ('C2', 'N3', 'C4'), ('N3', 'C4', 'C5'), ('C4', 'C5', 'C6'),
                  ('C5', 'C6', 'N1'), ('N3', 'C4', 'N9'), ('C6', 'C5', 'N7'), ('C5', 'C4', 'N9'), ('C4', 'N9', 'C8'),
                  ('N9', 'C8', 'N7'), ('C8', 'N7', 'C5'), ('N7', 'C5', 'C4'), ('O6', 'C6', 'N1'), ('O6', 'C6', 'C5'),
                  ('N2', 'C2', 'N1'), ('N2', 'C2', 'N3')]
CYTOSINE_BONDS = [('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C2', 'O2'),
                  ('C4', 'N4')]
CYTOSINE_ANGLES = [('C6', 'N1', 'C2'), ('N1', 'C2', 'N3'), ('C2', 'N3', 'C4'), ('N3', 'C4', 'C5'), ('C4', 'C5', 'C6'),
                   ('C5', 'C6', 'N1'), ('O2', 'C2', 'N1'), ('O2', 'C2', 'N3'), ('N4', 'C4', 'C5'), ('N4', 'C4', 'N3')]
THYMINE_BONDS = [('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C2', 'O2'),
                 ('C4', 'O4'), ('CM', 'C5')]
THYMINE_ANGLES = [('C6', 'N1', 'C2'), ('N1', 'C2', 'N3'), ('C2', 'N3', 'C4'), ('N3', 'C4', 'C5'), ('C4', 'C5', 'C6'),
                  ('C5', 'C6', 'N1'), ('O2', 'C2', 'N1'), ('O2', 'C2', 'N3'), ('O4', 'C4', 'C5'), ('O4', 'C4', 'N3'),
                  ('CM', 'C5', 'C4'), ('CM', 'C5', 'C6')]
URACIL_BONDS = [('N1', 'C2'), ('C2', 'N3'), ('N3', 'C4'), ('C4', 'C5'), ('C5', 'C6'), ('C6', 'N1'), ('C2', 'O2'),
                ('C4', 'O4')]
URACIL_ANGLES = [('C6', 'N1', 'C2'), ('N1', 'C2', 'N3'), ('C2', 'N3', 'C4'), ('N3', 'C4', 'C5'), ('C4', 'C5', 'C6'),
                 ('C5', 'C6', 'N1'), ('O2', 'C2', 'N1'), ('O2', 'C2', 'N3'), ('O4', 'C4', 'C5'), ('O4', 'C4', 'N3')]

DA_BONDS = DEOXYRIBOSE_PURINE_SUGAR_BONDS + PO4_BONDS + ADENINE_BONDS
DC_BONDS = DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + CYTOSINE_BONDS
DG_BONDS = DEOXYRIBOSE_PURINE_SUGAR_BONDS + PO4_BONDS + GUANINE_BONDS
DT_BONDS = DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + THYMINE_BONDS
DU_BONDS = DEOXYRIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + URACIL_BONDS
A_BONDS = RIBOSE_PURINE_SUGAR_BONDS + PO4_BONDS + ADENINE_BONDS
C_BONDS = RIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + CYTOSINE_BONDS
G_BONDS = RIBOSE_PURINE_SUGAR_BONDS + PO4_BONDS + GUANINE_BONDS
T_BONDS = RIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + THYMINE_BONDS
U_BONDS = RIBOSE_PYRIMIDINE_SUGAR_BONDS + PO4_BONDS + URACIL_BONDS

ALL_BONDS_MAP = {
    'DA': DA_BONDS,
    'DC': DC_BONDS,
    'DG': DG_BONDS,
    'DT': DT_BONDS,
    'DU': DU_BONDS,
    'A': A_BONDS,
    'C': C_BONDS,
    'G': G_BONDS,
    'T': T_BONDS,
    'U': U_BONDS,
}

DA_ANGLES = DEOXYRIBOSE_PURINE_SUGAR_ANGLES + PO4_ANGLES + ADENINE_ANGLES
DC_ANGLES = DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + CYTOSINE_ANGLES
DG_ANGLES = DEOXYRIBOSE_PURINE_SUGAR_ANGLES + PO4_ANGLES + GUANINE_ANGLES
DT_ANGLES = DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + THYMINE_ANGLES
DU_ANGLES = DEOXYRIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + URACIL_ANGLES
A_ANGLES = RIBOSE_PURINE_SUGAR_ANGLES + PO4_ANGLES + ADENINE_ANGLES
C_ANGLES = RIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + CYTOSINE_ANGLES
G_ANGLES = RIBOSE_PURINE_SUGAR_ANGLES + PO4_ANGLES + GUANINE_ANGLES
T_ANGLES = RIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + THYMINE_ANGLES
U_ANGLES = RIBOSE_PYRIMIDINE_SUGAR_ANGLES + PO4_ANGLES + URACIL_ANGLES

ALL_ANGLES_MAP = {
    'DA': DA_ANGLES,
    'DC': DC_ANGLES,
    'DG': DG_ANGLES,
    'DT': DT_ANGLES,
    'DU': DU_ANGLES,
    'A': A_ANGLES,
    'C': C_ANGLES,
    'G': G_ANGLES,
    'T': T_ANGLES,
    'U': U_ANGLES,
}

RA_PO4_MAPPING = {
    "aOP1POP2": "aO1O2",
    "aOP1PO3'": "aO1O3",
    "aOP1PO5'": "aO1O5",
    "aOP2PO3'": "aO2O3",
    "aOP2PO5'": "aO2O5",
    "aO3'PO5'": "aO3O5",
    "aPO3'C3'": "aP4O3C3",
    "aPO5'C5'": "aP4O5C5",
    "dOP1P": "dO1P4",
    "dOP2P": "dO2P4",
    "dO3'P": "dO3P4",
    "dO5'P": "dO5P4",
    "dO3'C3'": "dO3C3",
    "dO5'C5'": "dO5C5",
}


def get_pdb_files(in_dir):
    return list(glob(os.path.join(in_dir, '*.pdb')))


def remove_disordered(chains):
    """
    Remove disordered residues form hierarchy.
    :param chains: pdb hierarchy
    :return: chains
    """
    to_remove = set()

    for chain_id, residues in chains.iteritems():
        for res_id, sites in residues.iteritems():
            for atom_label, alt_locs in sites.iteritems():
                for alt_loc, atom in alt_locs.iteritems():
                    if alt_loc != '':
                        to_remove.add((chain_id, res_id))

    for chain_id, res_id in to_remove:
        print('removing disordered chain {} resi {}'.format(chain_id, res_id))
        chains[chain_id].pop(res_id)
        if len(chains[chain_id]) == 0:
            chains.pop(chain_id)

    return chains


def construct_hierarchy(pdb, keep_only_res_names=None, keep_disordered=True):
    """
    :param pdb: ccdc.protein.Protein object
    :param keep_only_res_names: None or list or res_names to keep in hierarchy, for example ['A', 'DA']

    :return: chains, res_names_map
        chains: dict in form of {chain_id: {res_id: {atom_label: {alt_loc: atom}}}
        res_names_map dict in form of {chain_id: {res_id: res_name}}
    """
    chains = {}
    # chain_id:{res_id:res_name}
    res_names_map = {}

    for atom in pdb.atoms:
        annos = atom._atom.annotations()
        psd = annos.find_ProteinSubstructureData()

        chain_id = psd.chain_id().strip()
        res_id = psd.residue_sequence_number()
        res_name = psd.residue_name().strip()
        alt_loc = psd.alternate_location().strip()
        atom_label = atom.label.strip()

        if DEBUG:
            print(chain_id, res_name, res_id, atom_label, atom.atomic_symbol, alt_loc, atom.coordinates)

        # update resi names map
        res_names_map_for_chain = res_names_map.setdefault(chain_id, {})
        saved_res_name = res_names_map_for_chain.setdefault(res_id, res_name)

        if res_name != saved_res_name:
            print(chain_id, res_name, res_id, atom_label, atom.atomic_symbol, alt_loc, atom.coordinates)
            raise Exception('inconsistent res names in chain {} resi {} residue names {} and {}'.format(
                chain_id, res_id, res_name, saved_res_name
            ))

        if keep_only_res_names is None or res_name in keep_only_res_names:
            chain = chains.setdefault(chain_id, {})
            res = chain.setdefault(res_id, {})
            site = res.setdefault(atom_label, {})
            site[alt_loc] = atom

    if keep_disordered is False:
        chains = remove_disordered(chains)

    return chains, res_names_map


def calc_bond(bond_atom_names, residues, res_id):
    if len(bond_atom_names) == 2:
        try:
            atom0_xyz = residues[res_id][bond_atom_names[0]][''].coordinates
            atom1_xyz = residues[res_id][bond_atom_names[1]][''].coordinates
        except:
            return float('nan')
    else:
        try:
            atom0_xyz = residues[res_id + bond_atom_names[2]][bond_atom_names[0]][''].coordinates
            atom1_xyz = residues[res_id + bond_atom_names[3]][bond_atom_names[1]][''].coordinates
        except:
            return float('nan')

    dx = atom0_xyz.x-atom1_xyz.x
    dy = atom0_xyz.y-atom1_xyz.y
    dz = atom0_xyz.z-atom1_xyz.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def calc_restraintlib_bond(bond_atom_names, chain_id, res_id, restraints_df):
    bond_name = "d" + "".join(bond_atom_names[0:2])

    if len(bond_atom_names) > 2:
        res_id = res_id + bond_atom_names[2]

    val = restraints_df.loc[
        (restraints_df.restraint_name == bond_name) &
        (restraints_df.chain1 == chain_id) &
        (restraints_df.resi1 == res_id), "value"]

    if val.shape[0] > 0:
        return val.values[0]
    elif bond_name in RA_PO4_MAPPING:
        bond_name = RA_PO4_MAPPING[bond_name]
        val = restraints_df.loc[(restraints_df.restraint_name == bond_name) &
                                (restraints_df.chain1 == chain_id) &
                                (restraints_df.resi1 == res_id), "value"]
        if val.shape[0] > 0:
            return val.values[0]

    return np.nan


def calc_parkinson_bond(bond_atom_names, res_name, restraints_df):
    bond_name = "pb__" + "-".join(bond_atom_names[0:2])

    return restraints_df.loc[res_name, bond_name]


def calc_angle(angle_atom_names, residues, res_id):
    if len(angle_atom_names) == 3:
        try:
            atom0_xyz = residues[res_id][angle_atom_names[0]][''].coordinates
            atom1_xyz = residues[res_id][angle_atom_names[1]][''].coordinates
            atom2_xyz = residues[res_id][angle_atom_names[2]][''].coordinates
        except:
            return float('nan')
    else:
        try:
            atom0_xyz = residues[res_id + angle_atom_names[3]][angle_atom_names[0]][''].coordinates
            atom1_xyz = residues[res_id + angle_atom_names[4]][angle_atom_names[1]][''].coordinates
            atom2_xyz = residues[res_id + angle_atom_names[5]][angle_atom_names[2]][''].coordinates
        except:
            return float('nan')

    v1_dx = atom0_xyz.x - atom1_xyz.x
    v1_dy = atom0_xyz.y - atom1_xyz.y
    v1_dz = atom0_xyz.z - atom1_xyz.z

    v2_dx = atom2_xyz.x - atom1_xyz.x
    v2_dy = atom2_xyz.y - atom1_xyz.y
    v2_dz = atom2_xyz.z - atom1_xyz.z

    dot = v1_dx*v2_dx + v1_dy*v2_dy + v1_dz*v2_dz
    length_v1 = math.sqrt(v1_dx*v1_dx + v1_dy*v1_dy + v1_dz*v1_dz)
    length_v2 = math.sqrt(v2_dx*v2_dx + v2_dy*v2_dy + v2_dz*v2_dz)

    angle = math.acos(dot / (length_v1*length_v2))
    return math.degrees(angle)

def calc_restraintlib_angle(angle_atom_names, chain_id, res_id, restraints_df):
    angle_name = "a" + "".join(angle_atom_names[0:3])

    if len(angle_atom_names) > 3:
        res_id = res_id + angle_atom_names[3]

    val = restraints_df.loc[(restraints_df.restraint_name == angle_name) &
                            (restraints_df.chain1 == chain_id) &
                            (restraints_df.resi1 == res_id), "value"]
    if val.shape[0] > 0:
        return val.values[0]
    elif angle_name in RA_PO4_MAPPING:
        angle_name = RA_PO4_MAPPING[angle_name]
        val = restraints_df.loc[(restraints_df.restraint_name == angle_name) &
                                (restraints_df.chain1 == chain_id) &
                                (restraints_df.resi1 == res_id), "value"]
        if val.shape[0] > 0:
            return val.values[0]

    return np.nan


def calc_parkinson_angle(angle_atom_names, res_name, restraints_df):
    angle_name = "pa__" + "-".join(angle_atom_names[0:3])

    return restraints_df.loc[res_name, angle_name]


def calc_diffs(combined_df, prefixes, names, results_dir, pdb_code=None):
    diff_arrays = []
    isnotnan_arrays = []
    filtered_diff_arrays = []

    if pdb_code is not None:
        combined_df = combined_df.loc[combined_df.pdb_code == pdb_code, :]

    for prefix in prefixes:
        against_df = combined_df.loc[:, combined_df.columns.str.startswith(prefix)]
        diff_df = against_df.values - combined_df.loc[:, against_df.columns.str.lstrip(prefix)].values

        abs_diff_array = np.abs(diff_df.flatten())
        diff_arrays.append(abs_diff_array)
        isnotnan_arrays.append(~np.isnan(abs_diff_array))

    for idx, name in enumerate(names):
        filtered_df = diff_arrays[idx][np.all(isnotnan_arrays, axis=0)]
        filtered_diff_arrays.append(filtered_df)
        np.savetxt(os.path.join(results_dir, name + ".csv"), filtered_df, delimiter=",")

    return tuple(filtered_diff_arrays)


def rmsd(numpy_array):
    return (np.sum(numpy_array ** 2) / numpy_array.size) ** 0.5


def calc_bonds_and_angles(pdb_data, pdb_res_names_map, bonds_map, angles_map, data_dir, results_dir):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    data_bonds = defaultdict(list)
    data_restrailtlib_bonds = defaultdict(list)
    data_parkinson_bonds = defaultdict(list)
    data_angles = defaultdict(list)
    data_restraintlib_angles = defaultdict(list)
    data_parkinson_angles = defaultdict(list)
    data_names = defaultdict(list)

    parkinson_restraints_df = pd.read_csv("parkinson_restraints.csv", index_col=0)

    for pdb_code, chains in pdb_data.iteritems():
        restraintlib_restraints_df = pd.read_csv(os.path.join(data_dir, pdb_code + ".txt"), comment="#")

        for chain_id, residues in chains.iteritems():
            for res_id, res in residues.iteritems():

                res_name = pdb_res_names_map[pdb_code][chain_id][res_id]

                bonds = bonds_map[res_name]
                angles = angles_map[res_name]

                bonds_data = {"-".join(bond[0:2]): calc_bond(bond, residues, res_id) for bond in bonds}
                bonds_restraintlib_data = {("rb__" + "-".join(bond[0:2])): calc_restraintlib_bond(bond, chain_id, res_id, restraintlib_restraints_df) for bond in bonds}
                bonds_parkinson_data = {("pb__" + "-".join(bond[0:2])): calc_parkinson_bond(bond, res_name, parkinson_restraints_df) for bond in bonds}
                angle_data = {"-".join(angle[0:3]): calc_angle(angle, residues, res_id) for angle in angles}
                angle_restraintlib_data = {("ra__" + "-".join(angle[0:3])): calc_restraintlib_angle(angle, chain_id, res_id, restraintlib_restraints_df) for angle in angles}
                angle_parkinson_data = {("pa__" + "-".join(angle[0:3])): calc_parkinson_angle(angle, res_name, parkinson_restraints_df) for angle in angles}

                data_bonds[res_name].append(bonds_data)
                data_restrailtlib_bonds[res_name].append(bonds_restraintlib_data)
                data_parkinson_bonds[res_name].append(bonds_parkinson_data)
                data_angles[res_name].append(angle_data)
                data_restraintlib_angles[res_name].append(angle_restraintlib_data)
                data_parkinson_angles[res_name].append(angle_parkinson_data)
                data_names[res_name].append({'pdb_code': pdb_code, 'chain': chain_id, "resi": res_id, "res_name": res_name})

    combined_data = None

    for res_name in data_bonds.keys():
        names = pd.DataFrame(data_names[res_name])

        bonds = pd.DataFrame(data_bonds[res_name])
        restraintlib_bonds = pd.DataFrame(data_restrailtlib_bonds[res_name])
        parkinson_bonds = pd.DataFrame(data_parkinson_bonds[res_name])

        angles = pd.DataFrame(data_angles[res_name])
        restraintlib_angles = pd.DataFrame(data_restraintlib_angles[res_name])
        parkinson_angles = pd.DataFrame(data_parkinson_angles[res_name])

        data = pd.concat([names, bonds, angles, restraintlib_bonds, restraintlib_angles, parkinson_bonds, parkinson_angles], axis=1)
        data.to_csv(os.path.join(results_dir, "pdb_stats_{}.csv".format(res_name)), index=False)

        if combined_data is None:
            combined_data = data.copy()
        else:
            combined_data = pd.concat([combined_data, data], axis=0, ignore_index=True)

    combined_data.to_csv(os.path.join(results_dir, "pdb_stats_combined.csv"), index=False)

    summarize_results(combined_data, results_dir)


def summarize_results(combined_data, results_dir):
    diff_3p4j_parkinson_bonds, diff_3p4j_restraintlib_bonds = calc_diffs(combined_data, ['pb__', 'rb__'],
                                                                         ["3p4j_bonds_diff_parkinson",
                                                                          "3p4j_bonds_diff_restraintlib"], results_dir,
                                                                         pdb_code="3p4j")
    diff_3p4j_parkinson_angles, diff_3p4j_restraintlib_angles = calc_diffs(combined_data, ['pa__', 'ra__'],
                                                                           ["3p4j_angles_diff_parkinson",
                                                                            "3p4j_angles_diff_restraintlib"],
                                                                           results_dir, pdb_code="3p4j")
    diff_1d8g_parkinson_bonds, diff_1d8g_restraintlib_bonds = calc_diffs(combined_data, ['pb__', 'rb__'],
                                                                         ["1d8g_bonds_diff_parkinson",
                                                                          "1d8g_bonds_diff_restraintlib"], results_dir,
                                                                         pdb_code="1d8g")
    diff_1d8g_parkinson_angles, diff_1d8g_restraintlib_angles = calc_diffs(combined_data, ['pa__', 'ra__'],
                                                                           ["1d8g_angles_diff_parkinson",
                                                                            "1d8g_angles_diff_restraintlib"],
                                                                           results_dir, pdb_code="1d8g")
    diff_parkinson_bonds, diff_restraintlib_bonds = calc_diffs(combined_data, ['pb__', 'rb__'],
                                                               ["pdb_bonds_diff_parkinson",
                                                                "pdb_bonds_diff_restraintlib"], results_dir)
    diff_parkinson_angles, diff_restraintlib_angles = calc_diffs(combined_data, ['pa__', 'ra__'],
                                                                 ["pdb_angles_diff_parkinson",
                                                                  "pdb_angles_diff_restraintlib"], results_dir)

    summary = """-------------------------
3P4J [bonds: {:d}/{:d}, angles: {:d}/{:d}]
-------------------------
RMSD (bonds) Parkinson: {:.4f}
RMSD (bonds) Ours: {:.4f}
RMSD (angles) Parkinson: {:.4f}
RMSD (angles) Ours: {:.4f}
-------------------------
1D8G [bonds: {:d}/{:d}, angles: {:d}/{:d}]
-------------------------
RMSD (bonds) Parkinson: {:.4f}
RMSD (bonds) Ours: {:.4f}
RMSD (angles) Parkinson: {:.4f}
RMSD (angles) Ours: {:.4f}
-------------------------
Overall [bonds: {:d}/{:d}, angles: {:d}/{:d}]
-------------------------
RMSD (bonds) Parkinson: {:.4f}
RMSD (bonds) Ours: {:.4f}
RMSD (angles) Parkinson: {:.4f}
RMSD (angles) Ours: {:.4f}
-------------------------
Wilcoxon (bonds): {:.4f}
Wilcoxon (angles): {:.4f}
-------------------------""".format(diff_3p4j_parkinson_bonds.size, diff_3p4j_restraintlib_bonds.size,
                                    diff_3p4j_parkinson_angles.size, diff_3p4j_restraintlib_angles.size,
                                    rmsd(diff_3p4j_parkinson_bonds), rmsd(diff_3p4j_restraintlib_bonds),
                                    rmsd(diff_3p4j_parkinson_angles), rmsd(diff_3p4j_restraintlib_angles),

                                    diff_1d8g_parkinson_bonds.size, diff_1d8g_restraintlib_bonds.size,
                                    diff_1d8g_parkinson_angles.size, diff_1d8g_restraintlib_angles.size,
                                    rmsd(diff_1d8g_parkinson_bonds), rmsd(diff_1d8g_restraintlib_bonds),
                                    rmsd(diff_1d8g_parkinson_angles), rmsd(diff_1d8g_restraintlib_angles),

                                    diff_parkinson_bonds.size, diff_restraintlib_bonds.size,
                                    diff_parkinson_angles.size, diff_restraintlib_angles.size,
                                    rmsd(diff_parkinson_bonds),rmsd(diff_restraintlib_bonds),
                                    rmsd(diff_parkinson_angles),rmsd(diff_restraintlib_angles),

                                    wilcoxon(diff_parkinson_bonds, diff_restraintlib_bonds)[1],
                                    wilcoxon(diff_parkinson_angles, diff_restraintlib_angles)[1])
    print(summary)
    with open(os.path.join(results_dir, "summary.txt"), "w") as text_file:
        text_file.write(summary)

    simple_histogram([diff_restraintlib_angles, diff_parkinson_angles], 12, (0, 6), ["#77AADD", "#CCCCCC"],
                     ["This work", "Parkinson et al."], "Angle difference [$^\circ$]", "Count", results_dir,
                     "restrain_angle_comparison.hist")
    simple_histogram([diff_restraintlib_bonds, diff_parkinson_bonds], 12, (0, 0.06), ["#77AADD", "#CCCCCC"],
                     ["This work", "Parkinson et al."], "Bond length difference [$\AA$]", "Count", results_dir,
                     "restrain_dist_comparison.hist")


def calculate(data_dir, results_dir, bonds_map, angles_map):
    pdb_data = {}
    pdb_res_name_map = {}

    for pdb_file_name in get_pdb_files(data_dir):
        pdb_code = os.path.basename(pdb_file_name).replace('.pdb', '').lower()
        print('Reading', pdb_code)

        atom_count = 0
        with open(pdb_file_name, 'r') as pdf_file:
            for line in pdf_file.read().splitlines():
                if line.startswith('ATOM ') or line.startswith('HETATM'):
                    atom_count += 1

        pdb = Protein.from_file(pdb_file_name)

        if len(pdb.atoms) != atom_count:
            raise Exception('pdb_code {} inconsitent atom count self check: {} atoms: {}'.format(pdb_code, atom_count, len(pdb.atoms)))

        if DEBUG:
            print(pdb.identifier)
            print('chains count:', pdb._protein_structure.nchains())
            print('ligands count:', pdb._protein_structure.nligands())

            for chain in pdb.chains:
                print('{} Chain {} has {} residues'.format(pdb_code, chain.identifier, len(chain.residues)))


        chains, res_names_map = construct_hierarchy(pdb, keep_only_res_names=KEEP_RESIDUES, keep_disordered=KEEP_DISORDERED)
        # usually chains are not defined properly !!!
        # try to recover the hierarchy from atoms

        if DEBUG:
            for chain_id, residues in chains.iteritems():
                for res_id, res in residues.iteritems():
                    for atom_label, alt_locs in res.iteritems():
                        for alt_loc, atom in alt_locs.iteritems():
                            print(chain_id,
                                  res_names_map[chain_id][res_id],
                                  res_id,
                                  atom_label,
                                  atom.atomic_symbol,
                                  alt_loc,
                                  atom.coordinates)

        pdb_data[pdb_code] = chains
        pdb_res_name_map[pdb_code] = res_names_map

    calc_bonds_and_angles(pdb_data, pdb_res_name_map, bonds_map, angles_map, data_dir, results_dir)


def simple_histogram(data, bins, range, colors, labels, x_label, y_label, result_folder, result_filename):
    sns.set(style="ticks", palette="Set1", font_scale=1.5, font="Times New Roman")
    fg = plt.figure()
    plt.hist(data, bins, range, color=colors, label=labels)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    print("Saving histogram to:", result_filename)
    fg.savefig(os.path.join(result_folder, result_filename) + ".png", bbox_inches="tight", dpi=600)
    fg.savefig(os.path.join(result_folder, result_filename) + ".svg", format="svg", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    download_from_csv_file(csv="ndb_query_results.csv", out_dir="data", force=False, restraints=False)
    calculate("data", "results_sugars", SUGARS_BONDS_MAP, SUGARS_ANGLES_MAP)
