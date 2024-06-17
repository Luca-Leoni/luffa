"""Script to rewrite the ML_AB of VASP into an .xyz file"""

# ---- IMPORTS

# Numpy
import numpy as np

# ASE
from ase.io import write
from ase.stress import voigt_6_to_full_3x3_stress

# Typing
from typing import List
from ase import Atoms
from argparse import ArgumentParser, Namespace
import os


# ---- FUNCTIONS


def read_MLAB(filepath: str) -> List[Atoms]:
    """Read ML_AB from filepath and return it"""
    data: list[Atoms] = []

    with open(filepath, "r") as mlab:
        n: int = 0

        line = mlab.readline()
        while line:
            if "number of configurations" in line:
                mlab, line = skiprows(mlab, line, 2)
                n = int(line)

            # enter structure loop:
            if "Configuration num." in line:
                mlab, line = skiprows(mlab, line, 8)
                n_types = int(line)
                mlab, line = skiprows(mlab, line, 4)
                n_atoms = int(line)
                mlab, line = skiprows(mlab, line, 4)

                # read atom types
                types = []
                for _ in range(n_types):
                    types.extend([line.split()[0] for _ in range(int(line.split()[1]))])
                    line = mlab.readline()

                # read cell
                cell = np.loadtxt(mlab, skiprows=6, max_rows=3)

                # read atoms
                posi = np.loadtxt(mlab, skiprows=3, max_rows=n_atoms)

                # read energy
                mlab, line = skiprows(mlab, line, 4)
                energy = float(line)

                # read forces
                forc = np.loadtxt(mlab, skiprows=3, max_rows=n_atoms)

                # read stress
                mlab, line = skiprows(mlab, line, 6)
                stre = [float(x) for x in line.split()]

                mlab, line = skiprows(mlab, line, 4)
                stre.extend([float(x) for x in line.split()])

                # stre = np.array(stre)
                stre = voigt_6_to_full_3x3_stress(np.array(stre))

                # Add atom
                atoms = Atoms(types, posi, cell=cell, pbc=(True, True, True))

                atoms.info["energy"] = energy
                atoms.info["stress"] = stre

                atoms.arrays["forces"] = forc

                data.append(atoms)

            # end of structure reading
            mlab, line = skiprows(mlab, line, 1)
    # end data reading

    # Sanity check
    if (n == 0) or (len(data) != n):
        print("Something went wrong :(")
        print("Data shape does not match # of configurations.")
        exit(1)
    return data


def parse_arg() -> Namespace:
    """Get paths to input and output file
    as well as the output format from args."""
    parser = ArgumentParser(prog="rewrite_ML_AB")

    parser.add_argument(
        "ML_AB_path",
        help="path to ML_AB, if not supplied it is assumed to be ./ML_AB",
        default="ML_AB",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="path/name of output. If not supplied it will be ./data.<ext>, where the extension <ext> is either .pckl.gzip or .xyz depending on FORMAT",
        default="data",
    )

    return parser.parse_args()


def skiprows(file_obj, line, i) -> tuple:
    """Skip i lines with readline."""
    for i in range(i):
        line = file_obj.readline()
    return file_obj, line


# ---- MAIN


def main() -> None:
    """Save structures in ML_AB as specified file.
    Currently supporting pickeled DataFrames for pacemaker,
    and extended XYZ files for MACE/CACE."""
    # get arguments
    args = parse_arg()

    # sanity checks
    if not os.path.isfile(args.ML_AB_path):
        print(f"Could not find file: {args.ML_AB_path}")
        exit(1)

    # Read data
    data = read_MLAB(args.ML_AB_path)

    # write Data
    write(f"{args.out}.xyz", data, format="extxyz")


if __name__ == "__main__":
    main()
