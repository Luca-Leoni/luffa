from ..md import VaspMDAnalyzer

from time import time
from argparse import ArgumentParser
from numpy import save

parser = ArgumentParser(
    prog="Compute MSD",
    description="Simple CLI that read an XDATCAR and evaluate the MSD",
)

parser.add_argument(
    "elements",
    nargs="?",
    type=str,
    help="Elements for which the MSD should be computed, if none is give than all of them will be used",
)

parser.add_argument(
    "-xp",
    "--xdatcar_path",
    type=str,
    default="XDATCAR",
    help="Path to the XDATCAR file to read from, if not inserted a file named XDATCAR will be searched in the current directory",
)

parser.add_argument(
    "-sc",
    "--start_conf",
    type=int,
    default=0,
    help="Number of configuration to skip while reading the XDATCAR",
)

parser.add_argument(
    "-n",
    "--num_conf",
    type=int,
    default=1_000_000,
    help="Total number of configuration to read from the XDATCAR",
)

parser.add_argument(
    "-p",
    "--potim",
    type=float,
    default=1,
    help="Value of the POTIM variable used in the simulation",
)

parser.add_argument(
    "-o",
    "--output",
    action="store_true",
    help="Write the corrected XDATCAR",
)

args = parser.parse_args()


# REAL APPLICATION
def main():
    anal = VaspMDAnalyzer(
        args.xdatcar_path,
        args.potim,
        start_conf=args.start_conf,
        nconf=args.num_conf,
    )

    elements = args.elements if args.elements is not None else anal.get_atomic_species()

    for species in elements:
        print(f"\nCompute MSD for {species}:")
        start = time()
        save(f"MSD_{species}", anal.get_MSD(species, "fft"))
        print(f"Finished in {time() - start:.3f}s")

        anal.plot_MSD(species, show=True)

    if args.output:
        anal.write("./XDATCAR_out")
