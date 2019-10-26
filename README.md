# Conformation-dependent restraints for polynucleotides: The sugar moiety

Reproducible CSD analysis accompanying the paper "Conformation-dependent restraints for polynucleotides: The sugar moiety."

## Contents

The repository contains experimental source code, PDB validation data, image generation scripts, and detailed results of the analyses discussed in "Conformation-dependent restraints for polynucleotides: The sugar moiety." by Kowiel et al. The repository is divided into the following folders:

- the main folder contains the main soruce code files; to start the experiments run `sugar_analyses.py`
- `sugar_queries` contains CONQUEST queries used to retrieve structures with sugar fragments from the CSD
- `sugar_results` contains cached experiment results (these can be recreated by running `sugar_analyses.py`)
- `sugar_terminal_results` contains cached experiment results for terminal sugars
- `pdb` contains NDB validation data and script for calculating differences between bond lengths and angles
- the `refmac_runner`, `phenix_runner`, `refinement_analysis` contain additional code and results concerning refinements of structures with the use of the proposed restraints

## Requirements

To run the experiments the following software has to be installed:

- experiments were run using Python 2.7
- the main Python modules used were: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `plotnine`, `json`, `scikit-learn`
- CSD queries were run using the [CSD Python API](https://downloads.ccdc.cam.ac.uk/documentation/API/) version 2.1.0 and the `ccdc` module

## Contact

If you have trouble reproducing the experiments or have any comments/suggestions, feel free to write at dariusz.brzezinski (at) cs.put.poznan.pl