import numpy as np

from pymatgen.io.vasp.outputs import Xdatcar, Vasprun
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.core import Structure

from os import mkdir, chdir, system
from os.path import isfile, join, isdir

from argparse import ArgumentParser

import warnings

warnings.filterwarnings("ignore")


def __clean_INCAR(incar: str) -> str:
    wanted_flags = [
        "ENCUT",
        "ISMEAR",
        "SIGMA",
        "ALGO",
        "NELM",
        "EDIFF",
        "AMIN",
        "AMIX",
        "ISPIN",
        "IMIX",
    ]

    cleaned_incar = ""

    for flag in incar.splitlines():
        if flag.split()[0] in wanted_flags:
            cleaned_incar += flag + "\n"

    cleaned_incar += "LWAVE = .FALSE.\n"
    cleaned_incar += "LCHARG = .FALSE.\n"

    return cleaned_incar


def __run_dft():
    # Read files
    incar = __clean_INCAR(open("INCAR", "r").read())
    xdatcar = Xdatcar("XDATCAR").structures
    potcar = open("POTCAR", "r").read()
    kpoints = open("KPOINTS", "r").read()
    job = open("job.sh", "r").read()

    # select random structures
    rnd_idx = np.arange(len(xdatcar))
    np.random.shuffle(rnd_idx)
    rnd_idx = rnd_idx[:5]

    # Create file with selected structures
    with open("SELECTED_STRUCTURES.dat", "w") as file:
        file.write(
            "# File containing the index of the selected structure for the validation\n"
        )

        for n in rnd_idx:
            file.write(f"{n:4d}\n")

    # Create a folder to store all the data
    mkdir("DFT_DATA")
    chdir("DFT_DATA")

    for i, idx in enumerate(rnd_idx):
        folder = f"strut{i}"
        if not isdir(folder):
            mkdir(folder)

        # POSCAR
        Poscar(xdatcar[idx]).write_file(join(folder, "POSCAR"))

        # INCAR
        open(join(folder, "INCAR"), "w").write(incar)

        # KPOINTS
        open(join(folder, "KPOINTS"), "w").write(kpoints)

        # POTCAR
        open(join(folder, "POTCAR"), "w").write(potcar)

        # job
        open(join(folder, "job.sh"), "w").write(job)

        # Sbacth the stuff
        chdir(folder)
        system("sbatch job.sh")
        chdir("..")


def __chgnet_eval(
    structures: list[Structure], chgnet_path: str | None = None
) -> dict[str, list]:
    from chgnet.model.model import CHGNet

    if chgnet_path is None:
        model = CHGNet.load(verbose=False)
    else:
        model = CHGNet.from_file(chgnet_path)
    model.eval()

    chgnet_data: dict[str, list] = {
        "energies": [],
        "forces": [],
        "stresses": [],
    }

    ress = model.predict_structure(structures)

    for res in ress:
        chgnet_data["energies"].append(res["e"])  # pyright: ignore
        chgnet_data["forces"].append(res["f"])  # pyright: ignore
        chgnet_data["stresses"].append(res["s"])  # pyright: ignore

    return chgnet_data


def __vaspff_eval(str_idx) -> dict[str, list]:
    # Assumes a vasprun containing the data of the MLFF is present in the folder
    data = Vasprun(
        "vasprun.xml",
        parse_dos=False,
        parse_eigen=False,
        parse_potcar_file=False,
    ).ionic_steps

    vaspff_data: dict[str, list] = {
        "energies": [],
        "forces": [],
        "stresses": [],
    }

    N = len(data[0]["structure"])
    for idx in str_idx:
        vaspff_data["forces"].append(data[idx]["forces"])
        vaspff_data["energies"].append(data[idx]["e_0_energy"] / N)
        vaspff_data["stresses"].append(data[idx]["stress"])

    return vaspff_data


def __validate(
    chgnet: bool = True, vaspff: bool = True, chgnet_path: str | None = None
):
    # Structure used for validation
    str_idx = np.loadtxt("SELECTED_STRUCTURES.dat")

    # Get DFT data previously generated
    dft_data: dict[str, list] = {
        "energies": [],
        "forces": [],
        "stresses": [],
    }

    structures: list[Structure] = []

    for i in range(len(str_idx)):
        data = Vasprun(
            join("DFT_DATA", f"strut{i}", "vasprun.xml"),
            parse_dos=False,
            parse_eigen=False,
            parse_potcar_file=False,
        ).ionic_steps[0]

        structures.append(data["structure"])
        dft_data["forces"].append(data["forces"])
        dft_data["energies"].append(data["e_0_energy"] / len(data["structure"]))
        dft_data["stresses"].append(data["stress"])

    # Evaluation of other methods
    ml_data: dict[str, dict[str, list]] = dict()

    if chgnet:
        ml_data["chgnet"] = __chgnet_eval(structures, chgnet_path)
    if vaspff:
        ml_data["vaspff"] = __vaspff_eval(str_idx)

    # Evaluate MAE
    message = "Validation result:\n"
    for ml, data in ml_data.items():
        message += ml.upper() + " -> "
        for key in data.keys():
            mae = np.abs(np.array(data[key]) - np.array(dft_data[key])).mean()
            message += f"MAE {key}: {mae:.5f}     "
        message += "\n"

    print(message.strip())


parser = ArgumentParser(
    prog="Validation",
    usage="""
    Call it one time to outomatically read the XDATCAR, INCAR, POTCAR and KPOINTS files in the folder and sbatch a set of jobs 
    to perform static VASP computations on a set of structures randomly taken from the XDATCAR generating the DFT_DATA folder.
    If called again in a folder with DFT_DATA and SELECTED_STRUCTURES.dat in it will read the data obtained form VASP and 
    compute the MAE of energy, forces and stress between the selected ML model and VASP.
    The possible choices of the models are:
    - VASP ML Force Field (VASPFF): The code assumes that the XDATCAR was generated from a FF run whose vasprun.xml is present in the folder, taking the data from it.
    - CHGNet (CHGNET): The code will need the chgnet package to use the model in order to evaluate the results.
    """,
    description="Simple CLI used to perform validation of machine learning model for MD runs",
)

parser.add_argument(
    "-nf",
    "--number_of_structures",
    type=int,
    default=300,
    help="Number of structure to randomly extract from the XDATCAR to create the DFT database",
)

parser.add_argument(
    "-cg",
    "--chgnet",
    action="store_true",
    help="Tells that the validation should not be done against chgnet",
)

parser.add_argument(
    "-cgp",
    "--chgnet_path",
    type=str,
    default=None,
    help="Path to a store chgnet model to load during the validation",
)

parser.add_argument(
    "-vf",
    "--vaspff",
    action="store_true",
    help="Tells that the validation should not be done against vaspff",
)

args = parser.parse_args()


def main():
    if isdir("DFT_DATA") and isfile("SELECTED_STRUCTURES.dat"):
        __validate(not args.chgnet, not args.vaspff, args.chgnet_path)
    else:
        __run_dft()
