# ---- IMPORTS

# NUMPY
import numpy as np

# MATPLOTLIB
import matplotlib.pyplot as plt

# ASE
from ase.io import read

# Typing
from ase import Atoms
from argparse import ArgumentParser, Namespace


# ---- HELPER FUNCTION
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("xyz", help="Path to xyz file")

    return parser.parse_args()


# ---- MAIN
def main():
    args = arg_parse()

    traj = read(args.xyz, index=":", format="extxyz")

    if isinstance(traj, Atoms):
        traj = [traj]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    print(*traj[0].arrays)

    ax1.plot([atoms.arrays["magmoms"][:, -1] for atoms in traj])
    ax1.plot([atoms.arrays["magmoms"][:, -1].sum() for atoms in traj], color="black")

    charge = np.array([atoms.arrays["decomposed_charge"][:, -1] for atoms in traj])
    ax2.plot(charge)
    # ax2.plot([atoms.arrays["decomposed_charge"][:, -1].sum() for atoms in traj])

    fig.tight_layout()
    plt.show()
