"""Script for the parsing of the OUTCAR to obtain all the informations that one (Me, Luca Leoni) may want"""

# ---- IMPORT

# Numpy
import numpy as np

# Typing
from typing import Dict
from numpy import ndarray
from io import TextIOWrapper


# ---- FUNCTION


def go_to_match(file: TextIOWrapper, match: str, until: str = "") -> str:
    """
    Iterate through the File until a certain word is meatch in the line
    """
    line = file.readline()

    while line != "":
        if match in line:
            return line

        if (until in line) and (until != ""):
            break

        line = file.readline()
    return ""


def read_density_matrix(file: TextIOWrapper, lnumber: list[int]) -> tuple[list, list]:
    occup_up, occup_dw = [], []
    for ln in lnumber:
        go_to_match(file, "onsite density matrix")

        text = [file.readline() for _ in range(4 + 2 * ln)]

        occup_up.append(np.loadtxt(text[3:]))

        text = [file.readline() for _ in range(4 + 2 * ln)]

        occup_dw.append(np.loadtxt(text[3:]))

    return occup_up, occup_dw


def go_to_last_iteration(file: TextIOWrapper) -> str:
    # Search for the first iteration
    line = go_to_match(file, "Iteration")

    # See if already at the end of file
    if line == "":
        return line

    # Take the number of the sc loop
    numsc = int(line.split("(")[0].split()[-1])

    while line != "":
        # Save current position
        pos = file.tell() - len(line)

        # Reach next iteration
        line = go_to_match(file, "Iteration")
        # print(line)

        if line == "":
            # End of file reached

            file.seek(pos)
            break

        if numsc != int(line.split("(")[0].split()[-1]):
            # New sc loop started, return back and exit

            file.seek(pos)
            break

    return file.readline()


def is_unfinished(file: TextIOWrapper) -> bool:
    # save position
    pos = file.tell()

    if go_to_match(file, "LOOP+") != "":
        file.seek(pos)

        return False

    if go_to_match(file, "writing wavefunction") != "":
        file.seek(pos)

        return False

    return True


def parse_outcar(path: str, verbose: bool = False) -> Dict[str, ndarray]:
    """
    Parses the OUTCAR in order to obtain all dynamical informations about the run,
    is written in order to get all the steps if it's an MD run or only the last one if it's simple sc computation
    """
    data: Dict[str, list] = {}

    with open(path, "r") as f:
        # ---- ATOMIC SPECIES

        # Find atomic informations
        line, data["elements"] = go_to_match(f, "POTCAR"), []
        while "POTCAR" in line:
            data["elements"].append(line.split()[2].split("_")[0])

            line = f.readline()

        # Search for the number atoms per type
        line = go_to_match(f, "ions per type")
        nTypes = line.split("ions per type =")[-1].split()

        data["elements"] = [
            e for e, n in zip(data["elements"], nTypes) for _ in range(int(n))
        ]

        # Save total number of atoms
        nAtoms = len(data["elements"])

        # ---- DENSITY MATRIX INFORMATIONS
        line = go_to_match(f, "LDAUL", "Ionic step")

        lnumber = []
        if line != "":
            lnumber = [int(x) for x in line.split("LDAUL =")[-1].split()]
            lnumber = [x for x, n in zip(lnumber, nTypes) for _ in range(int(n))]

        # ---- SETUP FOR PARSING

        # Helper variables
        data["positions"], data["forces"], data["energies"] = [], [], []
        data["occup_up"], data["occup_dw"] = [], []
        data["magmom"], data["charge"] = [], []

        # MD helper variables
        data["tenergies"], data["temperature"] = [], []
        data["cells"], data["volume"] = [], []

        # ---- REAL PARSING

        # Run through the OUTCAR
        while True:
            # search for last iteration
            line = go_to_last_iteration(f)

            # Reached end
            if line == "":
                break

            # Check if this iteration is finished
            if is_unfinished(f):
                break

            # Print progres if all good
            if verbose:
                print(
                    f"Reading ionic step: {line.split("(")[0].split()[-1]:>6s}",
                    end="\r",
                )

            # Reading density matrix first
            occup_up, occup_dw = read_density_matrix(f, lnumber)
            data["occup_up"].append(occup_up)
            data["occup_dw"].append(occup_dw)

            # Reading total charge in the system
            line = go_to_match(f, "total charge\n")
            text = [f.readline() for _ in range(3 + nAtoms)]
            data["charge"].append(np.loadtxt(text[3:], usecols=(1, 2, 3, 4)))

            # Reading magmoms
            line = go_to_match(f, "magnetization (x)")
            text = [f.readline() for _ in range(3 + nAtoms)]
            data["magmom"].append(np.loadtxt(text[3:], usecols=(1, 2, 3, 4)))

            # If we want here that can be stress

            # Reading unit cell and volume
            line = go_to_match(f, "volume of cell")

            data["volume"].append(float(line.split(":")[-1]))

            text = [f.readline() for _ in range(4)]
            data["cells"].append(np.loadtxt(text[1:], usecols=(0, 1, 2)))

            # Reading positions and forces
            go_to_match(f, "POSITION")

            text = [f.readline() for _ in range(1 + nAtoms)]
            entries = np.loadtxt(text[1:])

            data["positions"].append(entries[:, :3])
            data["forces"].append(entries[:, 3:])

            # Reading FREE energy
            line = go_to_match(f, "TOTEN")
            data["energies"].append(float(line.split("=")[-1].split()[0]))

            # Reading temperature
            line = go_to_match(f, "temperature", "Ionic step")
            if line != "":
                data["temperature"].append(
                    float(line.split("temperature")[-1].split()[0])
                )

            # Reading TOTAL energy
            line = go_to_match(f, "ETOTAL", "Ionic step")
            if line != "":
                data["tenergies"].append(float(line.split("=")[-1].split()[0]))

    if verbose:
        print(f"Reading ionic step: {len(data["cells"]):>6d}")

    # ---- Transform and output
    return {key: np.array(item) for key, item in data.items()}


if __name__ == "__main__":
    data = parse_outcar(
        "/home/utente/MOUNT/DATA/HYB/MgO/DFT+U/PRODUCTION/2RUN/OUTCAR", True
    )

    for key, item in data.items():
        print(f"{key}: {len(item)}")
