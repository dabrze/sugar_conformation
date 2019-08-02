# coding: utf-8
# Authors: Marcin Kowiel <mkowiel@ump.edu.pl>
from __future__ import print_function

import urllib2
import os
import pandas as pd
from subprocess import Popen
import gzip


def download_pdb(pdb_name, out_dir, force):
    pdb_path = os.path.join(out_dir, pdb_name + '.pdb')

    if (force is True) or (not os.path.exists(pdb_path)):
        print('DOWNLOADING', pdb_name)
        # url = 'ftp://ftp.ebi.ac.uk/pub/databases/rcsb/pdb/data/structures/all/pdb/pdb%s.ent.gz' % pdb_name.lower()
        url = 'http://www.rcsb.org/pdb/files/%s.pdb' % pdb_name.upper()
        print(url)
        pdb = urllib2.urlopen(url)
        data = pdb.read()
        print("SAVING", pdb_path)
        with open(pdb_path, 'w') as pdb_out:
            pdb_out.write(data)


def download_from_query_file(files, out_dir, force):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for filename in files:
        pdb_file = open(filename, 'r')
        lines = pdb_file.read().splitlines()
        for line in lines:
            s_line = line.split(",")
            for pdb_name in s_line:
                pdb_name = pdb_name.strip()
                download_pdb(pdb_name, out_dir, force)


def download_from_csv_file(csv, out_dir, force, restraints):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    df = pd.read_csv(csv)
    for pdb_name in df.loc[:, "PDB ID"].unique():
        pdb_name = pdb_name.strip()
        download_pdb(pdb_name, out_dir, force)

        if restraints:
            p = Popen(['cctbx.python', '../../Restrains/restrains.py', 'Csv', os.path.join(out_dir, pdb_name + ".pdb"),
                       os.path.join(out_dir, pdb_name + ".txt")], shell=True)


# if __name__ == '__main__':
#     download_from_query_file(files=DEFAULT_QUERY_FILE, out_dir=OUT_DIR, force=True)
