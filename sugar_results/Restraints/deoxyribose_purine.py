DEOXYRIBOSE_PURINE_PDB_CODES = ['DA', 'DG']
DEOXYRIBOSE_PURINE_ALL_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_CHI_GAMMA_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_CHI_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_CONFORMATION_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_SUGAR_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_GAMMA_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_PDB_CODES = DEOXYRIBOSE_PURINE_PDB_CODES

DEOXYRIBOSE_PURINE_ATOM_NAMES = {
    "C1'": "C1'",
    "C1*": "C1'",
    "C2'": "C2'",
    "C2*": "C2'",
    "C3'": "C3'",
    "C3*": "C3'",
    "C4": "C4",
    "C4'": "C4'",
    "C4*": "C4'",
    "C5'": "C5'",
    "C5*": "C5'",
    "C8": "C8",
    "N9": "N9",
    "O3'": "O3'",
    "O3*": "O3'",
    "O4'": "O4'",
    "O4*": "O4'",
    "O5'": "O5'",
    "O5*": "O5'",
    "P": "P"
}
DEOXYRIBOSE_PURINE_ALL_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_CHI_GAMMA_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_CHI_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_CONFORMATION_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_SUGAR_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_GAMMA_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_ATOM_NAMES = DEOXYRIBOSE_PURINE_ATOM_NAMES

DEOXYRIBOSE_PURINE_ATOM_RES = {
    "C1'": 0,
    "C2'": 0,
    "C3'": 0,
    "C4": 0,
    "C4'": 0,
    "C5'": 0,
    "C8": 0,
    "N9": 0,
    "O3'": 0,
    "O4'": 0,
    "O5'": 0
}
DEOXYRIBOSE_PURINE_ALL_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_CHI_GAMMA_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_CHI_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_CONFORMATION_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_SUGAR_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_GAMMA_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_ATOM_RES = DEOXYRIBOSE_PURINE_ATOM_RES

DEOXYRIBOSE_PURINE_REQUIRED_CONDITION = [
    ("C1'", "C2'", 2.0, 0, 0),
    ("C2'", "C3'", 2.0, 0, 0),
    ("C3'", "C4'", 2.0, 0, 0),
    ("C4'", "O4'", 2.0, 0, 0),
    ("C1'", "O4'", 2.0, 0, 0),
    ("C3'", "O3'", 2.0, 0, 0),
    ("C4'", "C5'", 2.0, 0, 0),
    ("C5'", "O5'", 2.0, 0, 0),
    ("C1'", 'N9', 2.0, 0, 0),
    ("O5'", 'P', 2.5, 0, 0),
    ("O3'", 'P', 2.5, 0, 1)
]
DEOXYRIBOSE_PURINE_ALL_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_CHI_GAMMA_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_CHI_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_CONFORMATION_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_SUGAR_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_GAMMA_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_REQUIRED_CONDITION = DEOXYRIBOSE_PURINE_REQUIRED_CONDITION

DEOXYRIBOSE_PURINE_DISTANCE_MEASURE = {
    'measure': 'euclidean_angles',
    'restraint_names': ["aC4'C5'O5'", "aC4'C3'O3'", "aN9C1'C2'", "aC1'N9C4", "aC1'N9C8", "aN9C1'O4'", "aC2'C1'O4'", "aC2'C3'O3'", "aC1'C2'C3'", "aC2'C3'C4'", "aC3'C4'O4'", "aC1'O4'C4'", "aC3'C4'C5'", "aC5'C4'O4'"]
}
DEOXYRIBOSE_PURINE_ALL_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_GAMMA_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CONFORMATION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_SUGAR_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_GAMMA_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_DISTANCE_MEASURE

DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE = {
    'measure': 'euclidean_angles',
    'restraint_names': ["tO4'C1'N9C4", "tC3'C4'C5'O5'", "pC1'C2'C3'C4'O4'"]
}
DEOXYRIBOSE_PURINE_ALL_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_GAMMA_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CONFORMATION_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_SUGAR_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_CHI_CONFORMATION_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_GAMMA_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE
DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_CONDITION_DISTANCE_MEASURE = DEOXYRIBOSE_PURINE_CONDITION_DISTANCE_MEASURE

DEOXYRIBOSE_PURINE_ALL_RESTRAINTS = [{
        'conditions': [], 'name': 'deoxyribose_purine==All=All', 'restraints': [['dist', "dC1'C2'", ["C1'", "C2'"], 1.525, 0.012], ['dist', "dC2'C3'", ["C2'", "C3'"], 1.523, 0.011]]
    }
]

DEOXYRIBOSE_PURINE_CHI_GAMMA_RESTRAINTS = [
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 60, 8.75]],
        'name': 'deoxyribose_purine==Chi=anti__Gamma=gauche+',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 110.6, 1.9]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], -60, 8.75]],
        'name': 'deoxyribose_purine==Chi=anti__Gamma=gauche-',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 109.6, 1.8]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 180, 21.25]],
        'name': 'deoxyribose_purine==Chi=anti__Gamma=trans',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 110.2, 1.9]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 60, 8.75]],
        'name': 'deoxyribose_purine==Chi=syn__Gamma=gauche+',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 112.5, 1.9]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], -60, 8.75]],
        'name': 'deoxyribose_purine==Chi=syn__Gamma=gauche-',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 111.0, 0.9]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5], ['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 180, 21.25]],
        'name': 'deoxyribose_purine==Chi=syn__Gamma=trans',
        'restraints': [['angle', "aC4'C5'O5'", ["C4'", "C5'", "O5'"], 110.5, 2.3]]
    }
]

DEOXYRIBOSE_PURINE_CHI_RESTRAINTS = [
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5]],
        'name': 'deoxyribose_purine==Chi=anti',
        'restraints': [['angle', "aC4'C3'O3'", ["C4'", "C3'", "O3'"], 110.7, 2.3]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5]],
        'name': 'deoxyribose_purine==Chi=syn',
        'restraints': [['angle', "aC4'C3'O3'", ["C4'", "C3'", "O3'"], 109.8, 2.1]]
    }
]

DEOXYRIBOSE_PURINE_BASE_FUNC_OF_TORSION_CHI_RESTRAINTS = [
    {
        'conditions': [],
        'name': 'deoxyribose_purine==Base=purine',
        'restraints': [   ['angle', "aN9C1'C2'", ['N9', "C1'", "C2'"], None, None, None, None, "purine-N1-C1'-C2' or N9-C1'-C2'.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]],
                          ['angle', "aC1'N9C4", ["C1'", 'N9', 'C4'], None, None, None, None, "purine-C1'-N1-C2 or C1'-N9-C4.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]],
                          ['angle', "aC1'N9C8", ["C1'", 'N9', 'C8'], None, None, None, None, "purine-C1'-N1-C6 or C1'-N9-C8.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]],
                          ['angle', "aN9C1'O4'", ['N9', "C1'", "O4'"], None, None, None, None, "purine-N1-C1'-O4' or N9-C1'-O4'.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]]]
    }
]

DEOXYRIBOSE_PURINE_CONFORMATION_RESTRAINTS = [
    {
        'conditions': [['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 162, 4.5]],
        'name': "deoxyribose_purine==Conformation=C2'-endo",
        'restraints': [['dist', "dC3'C4'", ["C3'", "C4'"], 1.527, 0.01], ['angle', "aC2'C1'O4'", ["C2'", "C1'", "O4'"], 106.0, 0.8]]
    },
    {
        'conditions': [['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 18, 4.5]],
        'name': "deoxyribose_purine==Conformation=C3'-endo",
        'restraints': [['dist', "dC3'C4'", ["C3'", "C4'"], 1.52, 0.009], ['angle', "aC2'C1'O4'", ["C2'", "C1'", "O4'"], 107.3, 0.6]]
    },
    {
        'conditions': [],
        'name': 'deoxyribose_purine==Conformation=Other',
        'restraints': [['dist', "dC3'C4'", ["C3'", "C4'"], 1.531, 0.009], ['angle', "aC2'C1'O4'", ["C2'", "C1'", "O4'"], 106.2, 1.3]]
    }
]

DEOXYRIBOSE_PURINE_SUGAR_RESTRAINTS = [{
        'conditions': [], 'name': 'deoxyribose_purine==Sugar=deoxyribose', 'restraints': [['dist', "dC4'O4'", ["C4'", "O4'"], 1.445, 0.009]]
    }
]

DEOXYRIBOSE_PURINE_CHI_CONFORMATION_RESTRAINTS = [
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5], ['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 162, 4.5]],
        'name': "deoxyribose_purine==Chi=anti__Conformation=C2'-endo",
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 109.4, 2.4]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5], ['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 18, 4.5]],
        'name': "deoxyribose_purine==Chi=anti__Conformation=C3'-endo",
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 113.4, 2.1]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 180, 22.5]],
        'name': 'deoxyribose_purine==Chi=anti__Conformation=Other',
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 111.9, 2.5]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5], ['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 162, 4.5]],
        'name': "deoxyribose_purine==Chi=syn__Conformation=C2'-endo",
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 110.1, 2.2]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5], ['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 18, 4.5]],
        'name': "deoxyribose_purine==Chi=syn__Conformation=C3'-endo",
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 114.2, 0.9]]
    },
    {
        'conditions': [['torsion', "tO4'C1'N9C4", ["O4'", "C1'", 'N9', 'C4'], 0, 22.5]],
        'name': 'deoxyribose_purine==Chi=syn__Conformation=Other',
        'restraints': [['angle', "aC2'C3'O3'", ["C2'", "C3'", "O3'"], 113.0, 1.7]]
    }
]

DEOXYRIBOSE_PURINE_SUGAR_CONFORMATION_FUNC_OF_TAU_MAX_RESTRAINTS = [
    {
        'conditions': [['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 162, 4.5]],
        'name': "deoxyribose_purine==Sugar=deoxyribose__Conformation=C2'-endo",
        'restraints': [   ['angle', "aC1'C2'C3'", ["C1'", "C2'", "C3'"], None, None, None, None, "deoxyribose-C2'-endo-C1'-C2'-C3'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC2'C3'C4'", ["C2'", "C3'", "C4'"], None, None, None, None, "deoxyribose-C2'-endo-C2'-C3'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC3'C4'O4'", ["C3'", "C4'", "O4'"], None, None, None, None, "deoxyribose-C2'-endo-C3'-C4'-O4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC1'O4'C4'", ["C1'", "O4'", "C4'"], None, None, None, None, "deoxyribose-C2'-endo-C1'-O4'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]]]
    },
    {
        'conditions': [['pseudorotation', "pC1'C2'C3'C4'O4'", ["C1'", "C2'", "C3'", "C4'", "O4'"], 18, 4.5]],
        'name': "deoxyribose_purine==Sugar=deoxyribose__Conformation=C3'-endo",
        'restraints': [   ['angle', "aC1'C2'C3'", ["C1'", "C2'", "C3'"], None, None, None, None, "deoxyribose-C3'-endo-C1'-C2'-C3'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC2'C3'C4'", ["C2'", "C3'", "C4'"], None, None, None, None, "deoxyribose-C3'-endo-C2'-C3'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC3'C4'O4'", ["C3'", "C4'", "O4'"], None, None, None, None, "deoxyribose-C3'-endo-C3'-C4'-O4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC1'O4'C4'", ["C1'", "O4'", "C4'"], None, None, None, None, "deoxyribose-C3'-endo-C1'-O4'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]]]
    },
    {
        'conditions': [],
        'name': 'deoxyribose_purine==Sugar=deoxyribose__Conformation=Other',
        'restraints': [   ['angle', "aC1'C2'C3'", ["C1'", "C2'", "C3'"], None, None, None, None, "deoxyribose-Other-C1'-C2'-C3'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC2'C3'C4'", ["C2'", "C3'", "C4'"], None, None, None, None, "deoxyribose-Other-C2'-C3'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC3'C4'O4'", ["C3'", "C4'", "O4'"], None, None, None, None, "deoxyribose-Other-C3'-C4'-O4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]],
                          ['angle', "aC1'O4'C4'", ["C1'", "O4'", "C4'"], None, None, None, None, "deoxyribose-Other-C1'-O4'-C4'.joblib", ['tau_max', ["C1'", "C2'", "C3'", "C4'", "O4'"]]]]
    }
]

DEOXYRIBOSE_PURINE_GAMMA_RESTRAINTS = [
    {
        'conditions': [['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 60, 8.75]],
        'name': 'deoxyribose_purine==Gamma=gauche+',
        'restraints': [['dist', "dC4'C5'", ["C4'", "C5'"], 1.508, 0.009], ['angle', "aC3'C4'C5'", ["C3'", "C4'", "C5'"], 115.7, 1.2], ['angle', "aC5'C4'O4'", ["C5'", "C4'", "O4'"], 109.4, 1.0]]
    },
    {
        'conditions': [['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], -60, 8.75]],
        'name': 'deoxyribose_purine==Gamma=gauche-',
        'restraints': [['dist', "dC4'C5'", ["C4'", "C5'"], 1.518, 0.009], ['angle', "aC3'C4'C5'", ["C3'", "C4'", "C5'"], 114.5, 1.2], ['angle', "aC5'C4'O4'", ["C5'", "C4'", "O4'"], 107.8, 0.9]]
    },
    {
        'conditions': [['torsion', "tC3'C4'C5'O5'", ["C3'", "C4'", "C5'", "O5'"], 180, 21.25]],
        'name': 'deoxyribose_purine==Gamma=trans',
        'restraints': [['dist', "dC4'C5'", ["C4'", "C5'"], 1.509, 0.01], ['angle', "aC3'C4'C5'", ["C3'", "C4'", "C5'"], 113.8, 1.3], ['angle', "aC5'C4'O4'", ["C5'", "C4'", "O4'"], 109.9, 1.2]]
    }
]

DEOXYRIBOSE_PURINE_ALL_FUNC_OF_TORSION_CHI_RESTRAINTS = [
    {
        'conditions': [],
        'name': 'deoxyribose_purine==All=All',
        'restraints': [   ['dist', "dC1'N9", ["C1'", 'N9'], None, None, None, None, "All-C1'-N1 or C1'-N9.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]],
                          ['dist', "dC1'O4'", ["C1'", "O4'"], None, None, None, None, "All-C1'-O4'.joblib", ['torsion_chi', ["O4'", "C1'", 'N9', 'C4']]]]
    }
]