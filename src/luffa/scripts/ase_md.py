import numpy as np

from os.path import basename, isfile, dirname, join

from ase.io import read, write
from ase import units

from argparse import ArgumentParser, Namespace

from tqdm import tqdm


def arg_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "trajectory_path",
        help="Path to the ase trajectory file",
        type=str,
    )

    parser.add_argument("-m", "--mean", type=str, required=True)
    return parser.parse_args()


possible_means = {"e": 1, "t": 4, "v": 5, "p": 6}


def find_wanted_averages(mean_flag: str) -> list[str]:
    res = []
    for char in mean_flag:
        if char in possible_means.keys():
            res.append(char)

    return res


def main():
    args = arg_parse()

    # Retrive file name
    folder = dirname(args.trajectory_path)
    name = basename(args.trajectory_path).split(".")[0]

    # Read logs
    if isfile(join(folder, name + ".log")):
        # If present not too much effort is needed
        logs = np.loadtxt(join(folder, name + ".log"), skiprows=1)
    else:
        # Otherwise we need to read the whole trajectory
        traj = read(args.trajectory_path, index=":")

        if not isinstance(traj, list):
            traj = [traj]

        logs, time = [], 1.0
        for atoms in tqdm(traj):
            epot = atoms.get_potential_energy()
            ekin = atoms.get_kinetic_energy()
            temp = atoms.get_temperature()
            volu = atoms.get_volume()
            pres = (
                np.trace(atoms.get_stress(voigt=False, include_ideal_gas=True))
                / units.GPa
                / 3
                * 10
            )

            logs.append(
                [time / units.fs / 1000, epot + ekin, epot, ekin, temp, volu, -pres]
            )
        logs = np.array(logs)

    # Compute wanted averages
    averages = find_wanted_averages(args.mean)

    mean = dict()
    for key in averages:
        vals = logs[:, possible_means[key]]

        mean[key] = vals.mean()
        print(f"Selecting structure with {key} = {mean[key]:.3f}")

    # Select s,tructures
    eps = 1e-5
    while True:
        idxs = np.arange(logs.shape[0])
        print(f"Try with relative tollerance of {eps:e}")
        for key in averages:
            vals = logs[:, possible_means[key]]

            idxs = idxs[np.isclose(vals[idxs], mean[key], rtol=eps)]

        if len(idxs) > 0:
            break
        else:
            eps *= 1.2

    print(f"Found the following structures: {idxs}")
    print("Picking the latest in the run...")

    # Print out the last structure with properties close to mean
    atoms = read(args.trajectory_path, index=idxs[-1])
    if isinstance(atoms, list):
        atoms = atoms[-1]

    write(join(folder, name + f"_{args.mean}.xyz"), atoms, format="extxyz")
    print(f"\nDONE: {name}_{args.mean}.xyz written")
