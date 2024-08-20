"""Script to automatically compute the polaron MSD from Leopold output"""

# ---- IMPORT

import numpy as np
import tables as tb

from ase.data import atomic_masses

from argparse import ArgumentParser, Namespace

import threading

# ---- HELPER FUNCTION


def fft_msd(atomic_positions: np.ndarray) -> np.ndarray:
    N = atomic_positions.shape[0]  # Number of frames

    D = np.square(atomic_positions)
    D1 = np.append(np.zeros_like(D[0:1]), D, 0)
    D2 = np.append(D, np.zeros_like(D[0:1]), 0)

    S1 = (2 * D.sum(0, keepdims=True) - np.cumsum(D1 + np.flip(D2, 0), 0))[:-1]

    S2 = np.fft.fft(atomic_positions, 2 * N, axis=0)
    S2 = np.fft.ifft(S2 * S2.conjugate(), axis=0)[:N].real

    return ((S1 - 2 * S2) / (N - np.arange(N).reshape(N, 1, 1))).mean(1)


def arg_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("file", type=str, help="Trajectory file to read")

    parser.add_argument(
        "-b",
        "--beg",
        type=int,
        default=0,
        help="Frame to use as a start for computing MSD",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="Frame to use as ending for computing MSD",
    )

    parser.add_argument(
        "-c",
        "--chunks",
        type=int,
        default=-1,
        help="Cut the trajectory in chunks and average over it, needed if the amount of data is too large",
    )

    parser.add_argument("-o", "--output", default="POL_MSD.npy")

    return parser.parse_args()


def compute_msd(msd, file, i, delta):
    print(f"Reading data between frames: {i * delta:<7d} ===> {(i + 1) * delta:<7d}")

    file = tb.open_file(file)

    # Read
    position = file.root.frames.read(i * delta, (i + 1) * delta, field="positions")
    pol_inde = file.root.frames.read(i * delta, (i + 1) * delta, field="polaron")
    cell = file.root.cell.read()
    mass = atomic_masses[np.int32(file.root.species.read())]

    file.close()

    print(f"Unwrapping data between frames: {i * delta:<7d} ===> {(i + 1) * delta:<7d}")

    # Unwrap the coordinates
    position = np.unwrap(position, axis=0, period=1)

    # Avoid drifting
    position -= (
        np.sum(position * mass[:, np.newaxis], axis=1, keepdims=True) / mass.sum()
    )

    # Take the polaron position
    position = np.sum(position * pol_inde[..., np.newaxis], axis=-2)

    # Unwrap polaron position
    position = np.unwrap(position, axis=0, period=1)

    # Transform coordinates to cartesian
    position = np.einsum("jk,kl->jl", position, cell)

    print(f"Compute MSD between frames: {i * delta:<7d} ===> {(i + 1) * delta:<7d}")

    # Compute the transform
    msd[i] = fft_msd(position[:, np.newaxis, :])


# ---- MAIN


def main():
    args = arg_parse()

    # Setting the end
    if args.end < 0:
        file = tb.open_file(args.file)

        args.end = file.root.frames.nrows + args.end + 1

        file.close()

    # Compute chunks values
    n_steps_per_chunk = (args.end - args.beg) // np.abs(args.chunks)

    msd = np.zeros((np.abs(args.chunks), n_steps_per_chunk, 3))

    # Serial version
    if args.chunks < 0:
        for i in range(np.abs(args.chunks)):
            compute_msd(msd, args.file, i, n_steps_per_chunk)
        np.save(args.output, msd.mean(0))

    # Parallel version
    else:
        ts = []

        for i in range(args.chunks):
            t = threading.Thread(
                target=compute_msd, args=(msd, args.file, i, n_steps_per_chunk)
            )
            t.start()

            ts.append(t)

        for t in ts:
            t.join()

        np.save(args.output, msd.mean(0))


if __name__ == "__main__":
    main()
