from pymatgen.io.vasp.outputs import Vasprun
from os import listdir
from os.path import isfile, join


def get_data_from_vasprun(
    file_root: str, check_electronic_convergence: bool = True
) -> dict[str, list]:
    vasprun_orig = Vasprun(
        f"{file_root}",
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
        parse_potcar_file=False,
        exception_on_bad_xml=False,
    )

    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])

    dataset = {
        "structures": [],
        "energies": [],
        "forces": [],
        "magmoms": [],
        "stresses": None if "stress" not in vasprun_orig.ionic_steps[0] else [],
    }

    for ionic_step in vasprun_orig.ionic_steps:
        if (
            check_electronic_convergence
            and len(ionic_step["electronic_steps"]) >= vasprun_orig.parameters["NELM"]
        ):
            continue

        dataset["structures"].append(ionic_step["structure"])
        dataset["energies"].append(ionic_step["e_0_energy"] / n_atoms)
        dataset["forces"].append(ionic_step["forces"])
        dataset["magmoms"].append([0 for _ in range(n_atoms)])
        if "stress" in ionic_step:
            dataset["stresses"].append(ionic_step["stress"])

    return dataset


def get_data_from_folder(
    folder_root: str, check_electronic_convergence: bool = True, check_name: bool = True
) -> dict[str, list]:
    files = listdir(folder_root)

    dataset: dict[str, list] = dict()
    for file in files:
        file_root = join(folder_root, file)

        if isfile(file_root):
            if not check_name:
                data = get_data_from_vasprun(file_root, check_electronic_convergence)
            elif file == "vasprun.xml":
                data = get_data_from_vasprun(file_root, check_electronic_convergence)
            else:
                continue
        else:
            data = get_data_from_folder(file_root, check_electronic_convergence)

        for key, value in data.items():
            if key not in dataset.keys():
                dataset[key] = []

            dataset[key].extend(value)

    return dataset
