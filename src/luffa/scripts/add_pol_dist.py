"""
Simple script to add polaronic like distorsions inside a POSCAR to a particular atom
"""

# ---- IMPORT

# MATH
import numpy as np

# OS
import os

# POSCAR
from pymatgen.io.vasp import Poscar

# Perturbation
from pymatgen.transformations.site_transformations import (
    RadialSiteDistortionTransformation,
)

# Types
from argparse import ArgumentParser, Namespace

# ---- FUNCTIONS


def args_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("path", help="path to the POSCAR file")
    parser.add_argument(
        "idxs",
        nargs="+",
        type=int,
        help="list of ion idxs to which add polaronic like distorsions",
    )

    parser.add_argument(
        "-r",
        "--pol_radius",
        type=float,
        default=0.5,
        help="value of the polaron radius defining the amplitude of lattice distorsions",
    )

    parser.add_argument("-o", "--output", default="POSCAR_POL")

    return parser.parse_args()


def main() -> None:
    args = args_parse()

    # Read POSCAR
    poscar = Poscar.from_file(args.path).structure

    # Modify structure
    for i in args.idxs:
        transf = RadialSiteDistortionTransformation(i, args.pol_radius, True)

        poscar = transf.apply_transformation(poscar)

    # Output
    output = "./POSCAR_POL"
    if args.output != "POSCAR_POL":
        output = args.output

    Poscar(poscar).write_file(output)


# ---- MAIN

if __name__ == "__main__":
    main()
