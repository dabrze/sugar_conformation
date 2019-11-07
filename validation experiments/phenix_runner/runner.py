import os
import shutil
from subprocess import Popen, PIPE

script = """refinement {{
  crystal_symmetry {{
    unit_cell = 60.4 60.4 133.1 90 90 120
    space_group = "P 61 2 2"
  }}
  input {{
    pdb {{
      file_name = {pdb_file}
    }}
    xray_data {{
      file_name = "{cif_file}"
      labels = "r429dsf,wavelength_id=1,_refln.F_meas_au,_refln.F_meas_sigma_au"
      r_free_flags {{
        file_name = "{cif_file}"
        label = "r429dsf,wavelength_id=1,_refln.status"
        test_flag_value = 0
      }}
    }}
  }}
  output {{
    prefix = "{job_prefix}"
    serial = {job_number}
    serial_format = "%d"
    job_title = "{job_title}"
    write_final_geo_file = True
    write_def_file = False
  }}
  electron_density_maps {{
    map_coefficients {{
      map_type = "2mFo-DFc"
      mtz_label_amplitudes = "2FOFCWT"
      mtz_label_phases = "PH2FOFCWT"
      fill_missing_f_obs = True
    }}
    map_coefficients {{
      map_type = "2mFo-DFc"
      mtz_label_amplitudes = "2FOFCWT_no_fill"
      mtz_label_phases = "PH2FOFCWT_no_fill"
    }}
    map_coefficients {{
      map_type = "mFo-DFc"
      mtz_label_amplitudes = "FOFCWT"
      mtz_label_phases = "PHFOFCWT"
    }}
    map_coefficients {{
      map_type = "anomalous"
      mtz_label_amplitudes = "ANOM"
      mtz_label_phases = "PHANOM"
    }}
  }}
  main {{
    number_of_macro_cycles = {number_of_cycles}
    nproc = Auto
  }}
  hydrogens {{
    refine = individual *riding Auto
  }}
  target_weights {{
    fix_wxc = {wxc}
  }}
  gui {{
    base_output_dir = "{output_dir}"
    tmp_dir = "{tmp_dir}"
    phil_file = {restraints_file}
  }}
}}
"""

config = {
    'pdb_file': 'data/429d.pdb',
    'cif_file': 'data/429d-sf.cif',
    'restraints_file': 'data/429d_sugars_restraints.txt',
    'output_dir': 'out/',
    'tmp_dir': 'tmp/',
    'number_of_cycles': 5,
    'job_prefix': '2GUN_refine',
    'job_title': '',
    'job_number': 0,
    'wxc': 3.0,
    'params_file': 'tmp/refine.params',
    'log': 'tmp/phenix.log',
    'phenix_path': 'phenix.refine.bat'
}

def run_once(script, config, run_log_filename):
    
    shell_cmd = '{phenix_path} {cif_file} {pdb_file} {params_file} --overwrite'
    
    cmd = shell_cmd.format(**config)
    cmd = cmd.split()

    script_ins = script.format(**config)
    with open(config['params_file'], 'w') as script_file:
        print >> script_file, script_ins

    p = Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE)
    stdout = p.communicate(input=script_ins)[0]
    std_out_log = stdout.decode()
    
    with open(config['log'], 'w') as log_file:
        print >> log_file, std_out_log
    
    i_find = std_out_log.find('        end:')
    if i_find > 0:
        log = std_out_log[i_find:]
        line = log.splitlines()[0]
        line = line.strip()
        values = line.split()

        r_work = values[1]
        r_free = values[2]
        rmsd_bonds = values[3]
        rmsd_angles = values[4]

        with open(run_log_filename, 'a') as run_log:
            print >> run_log, ";".join((str(_) for _ in (config['pdb_file'], r_work, r_free, rmsd_bonds, rmsd_angles, config['wxc'], config['restraints_file'])))


def run_value_change(script, config, config_keys, run_log_filename, init=0.0, step=0.05, size=11):

    with open(run_log_filename, 'w') as run_log:
        print >> run_log, ";".join((str(_) for _ in ('pdb_file', 'r_work', 'r_free', 'rmsd_bonds', 'rmsd_angles', 'wxc', 'restraints_file')))

    for i in range(size):
        value = init+step*i
        config['job_number'] = int(value*100)
        for config_key in config_keys:
            config[config_key] = value
        print 'run one', value
        run_once(script, config, run_log_filename)


def sugar_main(script, config):

    config['restraints_file'] = 'None'
    config['job_prefix'] = '2GUN_standard'

    if not os.path.exists('sugar_results/run_standard_429d.log'):
        run_value_change(script, config, ['wxc'], 'sugar_results/run_standard_429d.log', 6.0, 0.05, 11)

    config['restraints_file'] = '"data/429d_sugars_restraints.txt"'
    config['job_prefix'] = '2GUN_external'

    if not os.path.exists('sugar_results/run_external_429d.log'):
        run_value_change(script, config, ['wxc'], 'sugar_results/run_external_429d.log', 5.0, 0.05, 41)


sugar_main(script, config)


