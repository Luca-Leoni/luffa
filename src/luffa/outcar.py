"""Script for the parsing of the OUTCAR to obtain all the informations that one (Me, Luca Leoni) may want"""

# ---- IMPORT

# Numpy
import numpy as np

# Typing
from typing import Dict
from numpy import ndarray

# ---- FUNCTION


def parse_outcar(path: str, verbose: bool = False, l: int = 1) -> Dict[str, ndarray]:
    data: Dict[str, list] = {}

    with open(path, "r") as f:
        # Read the first line
        line = f.readline()

        # ---- STRUCT

        # Find atomic informations
        while "POSCAR" not in line:
            line = f.readline()

        data["elements"] = line.split("POSCAR:")[-1].split()

        # Search for the number atoms per type
        while "ions per type" not in line:
            line = f.readline()

        data["elements"] = [
            e
            for e, n in zip(data["elements"], line.split("ions per type =")[-1].split())
            for _ in range(int(n))
        ]

        # Save total number of atoms
        nAtoms = len(data["elements"])

        # Get the unit cell
        while "direct lattice vectors" not in f.readline():
            continue
        data["cell"] = np.loadtxt(f, max_rows=3)[:, :3].tolist()

        # ---- REAL PARSING

        # Set verbosity
        iter = f
        if verbose:
            from tqdm import tqdm

            iter = tqdm(f)

        # Helper variables
        occup_up, occup_dw = [], []
        data["positions"], data["forces"], data["energies"] = [], [], []
        data["occup_up"], data["occup_dw"] = [], []
        data["magmom"] = []

        # MD helper variables
        data["tenergies"], data["temperature"] = [], []
        data["cells"], data["volume"] = [], []

        # Run through the OUTCAR
        for line in iter:
            # Take occupation matrix, if present
            if "onsite density matrix" in line:
                # If toccup alread maxed out restart
                if len(occup_up) == nAtoms:
                    occup_up, occup_dw = [], []

                # Real read
                occup_up.append(np.loadtxt(f, skiprows=3, max_rows=2 * l + 1))
                occup_dw.append(np.loadtxt(f, skiprows=3, max_rows=2 * l + 1))

            # Take the possible unit cell and volume
            if "volume of cell" in line:
                data["volume"].append(float(line.split(":")[-1]))

                data["cells"].append(
                    np.loadtxt(f, skiprows=1, usecols=(0, 1, 2), max_rows=3)
                )

            # Take positions and forces
            if "POSITION" in line:
                entries = np.loadtxt(f, skiprows=1, max_rows=nAtoms)

                data["positions"].append(entries[:, :3])
                data["forces"].append(entries[:, 3:])

                # Final potential energy is written under the positions
                while "TOTEN" not in line:
                    line = f.readline()

                data["energies"].append(float(line.split("=")[-1].split()[0]))

                # Add toccup to final arrays since we reached the bottom
                data["occup_up"].append(occup_up)
                data["occup_dw"].append(occup_dw)

            # Save possible thermostat energy
            if "ETOTAL" in line:
                data["tenergies"].append(float(line.split("=")[-1].split()[0]))

                # After that ther's the temperature
                while "mean temperature" not in line:
                    line = f.readline()

                data["temperature"].append(float(line.split(":")[-1].split()[0]))

            # Take magnetization
            if "magnetization (x)" in line:
                data["magmom"].append(
                    np.loadtxt(f, skiprows=3, usecols=(1, 2, 3, 4), max_rows=nAtoms)
                )

    # Drop last MAGMOM and OCCUP since double counts
    if len(data["magmom"]) > 0:
        data["magmom"].pop()

    # Drop second unit cell since is a repetition
    if len(data["cells"]) > 1:
        data["cells"].pop(1)

    # ---- Transform and output
    return {key: np.array(item) for key, item in data.items()}
