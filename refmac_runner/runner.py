import os
import shutil
from subprocess import Popen, PIPE

script = """refi resolution {resolution_to} {resolution_from}
make check NONE
make -
    hydrogen ALL -
    hout NO -
    peptide NO -
    cispeptide YES -
    ssbridge YES -
    symmetry YES -
    sugar YES -
    connectivity NO -
    link NO
#NCSR LOCAL
refi -
    type REST -
    resi MLKF -
    meth CGMAT -
    bref ISOT
{tlsc_comment_out}refi tlsc {tlsc}
ncyc {ncyc}
scal -
    type BULK -
    reso {resolution_to} {resolution_from} -
    LSSC -
    ANISO -
    EXPE
solvent YES
weight matrix {weight_matrix}
#weight -
#    AUTO
monitor MEDIUM -
    torsion 10.0 -
    distance {monitor} -
    angle {monitor} -
    plane 10.0 -
    chiral 10.0 -
    bfactor 10.0 -
    bsphere 10.0 -
    rbond 10.0 -
    ncsr 10.0
labin  FP={FP} SIGFP={SIGFP} FREE={FREE}
labout  FC=FC FWT=FWT PHIC=PHIC PHWT=PHWT DELFWT=DELFWT PHDELWT=PHDELWT FOM=FOM
# only in final
#tlso addu
#dist 1.3     
#angle 1.57
vand -
    overall 3.54 -
    sigma metal 3.54
#temperature factors
#temp set 20.0
PNAME unknown
DNAME unknown190
RSIZE 80
# external
# bonds
EXTERNAL WEIGHT SCALE distance {external_distance}
EXTERNAL WEIGHT SCALE ANGLE {external_angle}
EXTERNAL WEIGHT GMWT {external_gmwt}
EXTERNAL USE ALL
EXTERNAL DMAX 4.2
# External script file:
{external_file}
END
## This script run with the command   ##########
# refmac5 XYZIN "3SSF.pdb" XYZOUT "100.pdb" HKLIN "hybrid-km-hamburg.mtz" HKLOUT "100.mtz" TLSIN "ref.tls" TLSOUT "100.tls" LIBOUT "100.cif"
################################################ 
"""

config = {
    'resolution_to': 50.0,
    'resolution_from': 1.95,
    'tlsc_comment_out': '#',
    'tlsc': 0,
    'ncyc': 30,
    'weight_matrix': 'AUTO', #0.08,
    'FP': 'FP',
    'SIGFP':'SIGFP', 
    'FREE': 'FREE',
    'external_distance': 3.0,
    'external_angle': 3.0,
    'external_gmwt': 0.5,
    'external_file': '',
    
    'xyzin': 'data/2han.pdb',
    'xyzout': 'tmp/100.pdb',
    'hklin': 'data/2han.mtz',
    'hklout': 'tmp/100.mtz',
    'TLSIN_arg': '',
    'tlsin': '',
    'TLSOUT_arg': '',
    'tlsout': '',
    'libout': 'tmp/100lib.cif',
    'monitor': 3.0,
    'log': 'tmp/100.log',
    'script_path': 'tmp/script.ins',
}

def run_once(script, config, run_log_filename):
    
    shell_cmd = 'refmac5 XYZIN {xyzin} XYZOUT {xyzout} HKLIN {hklin} HKLOUT {hklout} {TLSIN_arg} {tlsin} {TLSOUT_arg} {tlsout} LIBOUT {libout}'
    
    cmd = shell_cmd.format(**config)
    cmd = cmd.split()

    script_ins = script.format(**config)
    with open(config['script_path'], 'w') as script_file:
        print >> script_file, script_ins

    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout = p.communicate(input=script_ins)[0]
    std_out_log = stdout.decode()
    
    with open(config['log'], 'w') as log_file:
        print >> log_file, std_out_log
    
    i_find = std_out_log.find('$TEXT:Result: $$ Final results $$')
    if i_find > 0:
        log = std_out_log[i_find:]
        for line in log.splitlines():
            line = line.strip()

            if 'R factor' in line:
                sline = line.split()
                r_factor_init = sline[2]
                r_factor_final = sline[3]
            elif 'R free' in line:
                sline = line.split()
                r_free_init = sline[2]
                r_free_final = sline[3]
            elif 'Rms BondAngle' in line:
                sline = line.split()
                rms_bond_angle_init = sline[2]
                rms_bond_angle_final = sline[3]
            elif 'Rms BondLength' in line:
                sline = line.split()
                rms_bond_lenght_init = sline[2]
                rms_bond_lenght_final = sline[3]
            elif 'Rms ChirVolume' in line:
                sline = line.split()
                rms_chiral_volume_init = sline[2]
                rms_chiral_volume_final = sline[3]

        with open(run_log_filename, 'a') as run_log:
            print >> run_log, ";".join((str(_) for _ in (config['xyzin'], r_factor_init, r_free_init, r_factor_final, r_free_final, rms_bond_angle_final, rms_bond_lenght_final, rms_chiral_volume_final, config['weight_matrix'], config['ncyc'], config['tlsc'], config['external_distance'], config['external_angle'], config['external_gmwt'], config['external_file'],)))


def run_value_change(script, config, config_keys, run_log_filename, step=0.05, size=11):

    with open(run_log_filename, 'w') as run_log:
        print >> run_log, ";".join((str(_) for _ in ('xyzin', 'r_factor_init', 'r_free_init', 'r_factor_final', 'r_free_final', 'rms_bond_angle_final', 'rms_bond_lenght_final', 'rms_chiral_volume_final', 'weight_matrix', 'ncyc', 'tlsc', 'external_distance', 'external_angle', 'external_gmwt', 'external_file')))

    for value in (0.0+step*i for i in range(size)):
        for config_key in config_keys:
            config[config_key] = value
        print 'run one', value
        run_once(script, config, run_log_filename)


def sugar_main(script, config):

    config['weight_matrix'] = 0.00
    config['external_file'] = ''

    if not os.path.exists('sugar_results/run_standard_2han.log'):
        run_value_change(script, config, ['weight_matrix'], 'sugar_results/run_standard_2han.log', 0.01, 41)

    config['weight_matrix'] = 0.07
    config['external_gmwt'] = 0.5
    config['external_distance'] = 0
    config['external_angle'] = 0
    config['external_file'] = '@tmp/2han_restraints_refmac.in'

    if not os.path.exists('sugar_results/run_external_scale_2han.log'):
        run_value_change(script, config, ['external_distance', 'external_angle'], 'sugar_results/run_external_scale_2han.log', 0.1, 41)

    config['weight_matrix'] = 0.07
    config['external_gmwt'] = 0.5
    config['external_distance'] = 2.7
    config['external_angle'] = 2.7
    config['external_file'] = '@tmp/2han_restraints_refmac.in'

    if not os.path.exists('sugar_results/run_external_gmwt_2han.log'):
        run_value_change(script, config, ['external_gmwt'], 'sugar_results/run_external_gmwt_2han.log', 0.5, 11)

    if not os.path.exists('sugar_results/run_external_2han.log'):
        run_value_change(script, config, ['weight_matrix'], 'sugar_results/run_external_2han.log', 0.01, 41)


sugar_main(script, config)


