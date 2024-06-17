from ase import Atoms
from ase.utils import writer
from tqdm import tqdm


@writer
def write_xdatcar_from_traj(fd, images, verbose=True):
    """Write VASP MD trajectory (XDATCAR) file

    written in order to generate an output compatible with VASPKIT package for postprocessing analysis

    Args:
        fd (str, fp): Output file
        images (iterable of Atoms): Atoms images to write. These must have
            consistent atom order and lattice vectors - this will not be
            checked.
    """

    images = iter(images)
    image = next(images)

    if not isinstance(image, Atoms):
        raise TypeError("images should be a sequence of Atoms objects.")

    symbol_count = __symbol_count_from_symbols(image.get_chemical_symbols())

    __write_xdatcar_config(fd, image, symbol_count, 1)

    iteration = tqdm(enumerate(images)) if verbose else enumerate(images)
    for i, image in iteration:
        # Index is off by 2: 1-indexed file vs 0-indexed Python;
        # and we already wrote the first block.
        __write_xdatcar_config(fd, image, symbol_count, i + 2)


def __write_xdatcar_config(fd, atoms: Atoms, symbol_count, index):
    """Write a block of positions and lattice vectors for XDATCAR file

    Args:
        fd (fd): writeable Python file descriptor
        atoms (ase.Atoms): Atoms to write
        index (int): configuration number written to block header
    """
    # Not using lattice constants, set it to 1
    fd.write("unknown system\n")
    fd.write("           1\n")

    # Lattice vectors; use first image
    float_string = "{:11.6f}"
    for row_i in range(3):
        fd.write("  ")
        fd.write(" ".join(float_string.format(x) for x in atoms.cell.array[row_i]))
        fd.write("\n")

    __write_symbol_count(fd, symbol_count)
    fd.write("Direct configuration={:6d}\n".format(index))
    float_string = "{:11.8f}"
    scaled_positions = atoms.get_scaled_positions(wrap=False)
    for row in scaled_positions:
        fd.write(" ")
        fd.write(" ".join([float_string.format(x) for x in row]))
        fd.write("\n")


def __symbol_count_from_symbols(symbols):
    """Reduce list of chemical symbols into compact VASP notation

    args:
        symbols (iterable of str)

    returns:
        list of pairs [(el1, c1), (el2, c2), ...]
    """
    sc = []
    psym = symbols[0]
    count = 0
    for sym in symbols:
        if sym != psym:
            sc.append((psym, count))
            psym = sym
            count = 1
        else:
            count += 1
    sc.append((psym, count))
    return sc


def __write_symbol_count(fd, sc, vasp5=True):
    """Write the symbols and numbers block for POSCAR or XDATCAR

    Args:
        f (fd): Descriptor for writable file
        sc (list of 2-tuple): list of paired elements and counts
        vasp5 (bool): if False, omit symbols and only write counts

    e.g. if sc is [(Sn, 4), (S, 6)] then write::

      Sn   S
       4   6

    """
    if vasp5:
        for sym, _ in sc:
            fd.write(" {:3s}".format(sym))
        fd.write("\n")

    for _, count in sc:
        fd.write(" {:3d}".format(count))
    fd.write("\n")
