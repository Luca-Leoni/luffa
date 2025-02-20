"""Script to read an OUTCAR, normal or MD, and write an xyz file with all the informations"""

# ---- IMPORTS

# OUTCAR
from ..outcar import parse_outcar

# NUMPY
import numpy as np

# ASE
from ase.io import write

# Typing
from ase import Atoms
from argparse import ArgumentParser, Namespace


# ---- HELPER FUNCTION
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("outcar", help="Path to OUTCAR")

    # Options
    parser.add_argument("-o", "--output", default="outcar.xyz")
    parser.add_argument("-a", "--append", action="store_true")

    return parser.parse_args()


# ---- MAIN
def main():
    args = arg_parse()

    # Parse the OUTCAR
    data = parse_outcar(args.outcar, True)

    # Get the elements
    elements = data.pop("elements")

    # Pop all empty things
    keys = filter(lambda x: len(data[x]) == 0, list(data.keys()).copy())
    for key in keys:
        data.pop(key)

    # Create Atoms trajectory
    traj: list[Atoms] = []
    for (
        position,
        force,
        energy,
        occup_up,
        occup_dw,
        magmom,
        charge,
        tenergy,
        temp,
        cell,
        _,
    ) in zip(*data.values()):
        atoms = Atoms(elements, position, cell=cell, pbc=(True, True, True))

        atoms.info["energy"] = energy
        atoms.info["temperature"] = temp
        atoms.info["tenergy"] = tenergy

        atoms.arrays["forces"] = force
        atoms.arrays["charges"] = charge[:, -1]
        atoms.arrays["magmoms"] = magmom[:, -1]
        atoms.arrays["decomposed_charge"] = charge[:, :-1]
        atoms.arrays["decomposed_magmom"] = magmom[:, :-1]

        # See if present
        if len(occup_up) == 0:
            occup_up = 0.5 * (charge[:, :-1] - magmom[:, :-1])
            occup_dw = 0.5 * (charge[:, :-1] + magmom[:, :-1])

            atoms.arrays["toccups"] = np.append(occup_up, occup_dw, axis=1)
        else:
            atoms.arrays["toccups"] = np.dstack(
                (
                    np.trace(occup_dw, axis1=-1, axis2=-2),
                    np.trace(occup_up, axis1=-1, axis2=-2),
                )
            )[0]

        traj.append(atoms)

    write(args.output, traj, format="extxyz", append=args.append)


if __name__ == "__main__":
    main()
