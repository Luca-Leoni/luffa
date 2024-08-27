"""Script to perform a fit over an MSD numpy file"""

# ---- IMPORT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from argparse import ArgumentParser, Namespace
from ase.units import kB

# ---- HELPER FUNCTION


def arg_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("file", type=str, help="MSD file to read")

    parser.add_argument(
        "-b",
        "--beg",
        type=int,
        default=0,
        help="Starting point of the fit",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="last point of the fit",
    )

    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0,
        help="Temperature of the measurement, if inserted the Diffusion coeffienct is also transformed in a mobility",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save a picture with the plot of the MSD and the fit",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="MSD_FIT",
        help="Name of the picture output, if wanted",
    )

    return parser.parse_args()


# ---- MAIN


def main():
    args = arg_parse()

    # Load the data
    msd = np.load(args.file)

    data = msd[args.beg : args.end]
    time = np.arange(len(data))

    # Case where the MSD is computed for the cartesian coordinates
    if data.shape[1] == 3:
        # Compute the different coefficients
        Ds, Vs = np.zeros(4), np.zeros(4)

        Ds[:-1], Vs[:-1] = np.polyfit(time, data, 1)
        Ds[-1], Vs[-1] = np.polyfit(time, data.sum(-1), 1)

        # Print the results
        print(
            "D[A²/fs] %12.5E %10.5E %10.5E %10.5E"
            % (Ds[0] * 0.5, Ds[1] * 0.5, Ds[2] * 0.5, Ds[3] / 6)
        )

        if args.temperature > 0:
            mu = 10 * Ds / kB / args.temperature

            print(
                "μ[Vcm²/s] %10.5E %10.5E %10.5E %10.5E"
                % (mu[0] * 0.5, mu[1] * 0.5, mu[2] * 0.5, mu[3] / 6)
            )

        # Draw the final picture if wanted
        if args.save:
            colors = colormaps["magma"].reversed()(np.linspace(0, 1, 4))
            for i, label in enumerate(["X", "Y", "Z"]):
                plt.plot(msd[:, i], "--", label=label, color=colors[i])
            plt.plot(msd.sum(-1), label="Total", color=colors[-1], linewidth=3)

            for color, a, b in zip(colors, Ds, Vs):
                plt.plot(time, b + time * a, "-", color=color)

            plt.xlim(0, time[-1] * 1.1)
            plt.ylim(0, msd.sum(-1)[time[-1]] * 1.1)

            plt.legend()

            plt.tight_layout()
            plt.savefig(args.output + ".pdf")

    # Case where the MSD is computed for the projection
    else:
        # Compute the different coefficients
        Ds, _ = np.polyfit(time, data, 1)

        # Create the angle dependency
        theta = np.linspace(0, 2 * np.pi, len(Ds))

        # Save the results to file
        if args.temperature > 0:
            mu = 10 * Ds / kB / args.temperature

            np.save(args.output + ".npy", np.vstack((theta, 0.5 * mu, 0.5 * Ds)).T)
        else:
            np.save(args.output + ".npy", np.vstack((theta, 0.5 * Ds)).T)

        # Draw the final picture if wanted
        if args.save:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

            for i in range(5):
                ax.plot(Ds[i, :, 0] * 0.5, Ds[i, :, 1] * 0.5, label=f"{(i+1) * 100}K")

            plt.yscale("log")

            plt.ylabel(r"$\mu$")

            plt.legend()
            plt.savefig(args.output + ".pdf")


if __name__ == "__main__":
    main()
