from pathlib import Path
import numpy as np
import math
import tqdm
import openmm as mm
from openmm import LocalEnergyMinimizer
from openmm.app import Simulation
import openmm.unit as omm_unit
from openff.interchange import Interchange
from openff.units import unit as off_unit
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
from ase import Atoms
from ase.io import write as ase_write
from ase.constraints import FixInternals
from ase.optimize import BFGS
try:
    from xtb.ase.calculator import XTB  # GFN-xTB via ASE
except Exception:  # pragma: no cover
    XTB = None

try:
    from ase.calculators.psi4 import Psi4  # Psi4 via ASE
except Exception:  # pragma: no cover
    Psi4 = None

EV_TO_KCAL_MOL = 23.0609  # 1 eV = 23.0609 kcal/mol
EV_TO_KJ_MOL = 96.485  # 1 eV = 96.485 kJ/mol

def scan_dihedral_openmm(
    interchange: Interchange,
    dih_smarts: str,
    angles_deg=range(0, 361, 10),
    k_restraint: omm_unit.Quantity = 1000.0 * omm_unit.kilojoule_per_mole,
    platform_name: str = "Reference",
    minimize_tol: float = 1.0,
    max_iters: int = 10000,
    xyz_path: str | Path = "openff_scan.xyz",
    combine_nonbonded_forces: bool = True,
) -> pd.DataFrame:
    """Relaxed torsion scan using OpenMM + OpenFF Interchange.


    Steps
      1. Locate the four atoms via the provided SMARTS on the topology (RDKit) to get the dihedral indices.
      2. Update a PeriodicTorsionForce restraint to hold that angle.
      3. Minimize with ``Interchange.minimize(engine="openmm", ...)`` (updates positions in-place).
      4. Push minimized coords back to the OpenMM Context, record energy & write XYZ.
    Returns
    -------
    pd.DataFrame with columns ["angle", "energy_kJ_mol"].

    lots of problems with openmm but this apparently works for now :()
    """

    # --- utilities ---
    def _as_kjmol(val):
        """Return val as an OpenMM Quantity in kJ/mol."""
        if isinstance(val, omm_unit.Quantity):
            return val.value_in_unit(omm_unit.kilojoule_per_mole) * omm_unit.kilojoule_per_mole
        return val * omm_unit.kilojoule_per_mole

    def _to_nm(qty_array):
        """Convert an OpenMM Quantity array of positions to a unitless np.array in nm."""
        return qty_array.value_in_unit(omm_unit.nanometer)

    # 0) Resolve dihedral indices from SMARTS (required) using the OFF Molecule
    off_mols = list(interchange.topology.molecules)
    if len(off_mols) == 0:
        raise RuntimeError("No molecules found in Interchange topology.")
    if len(off_mols) > 1:
        # assume the first molecule unless user wants something fancier
        pass
    off_mol = off_mols[0]
    matches = off_mol.chemical_environment_matches(dih_smarts)
    if not matches:
        raise RuntimeError(f"No match for SMARTS '{dih_smarts}' in this topology.")
    dihedral_idx = tuple(matches[0])
    if len(dihedral_idx) != 4:
        raise RuntimeError(f"SMARTS must capture exactly 4 atoms; got {len(dihedral_idx)}.")

    # 1) Build OpenMM objects from Interchange
    system = interchange.to_openmm_system(combine_nonbonded_forces=combine_nonbonded_forces)  # no platform arg per API
    topo = interchange.topology.to_openmm()
    init_pos = interchange.positions.to_openmm()

    # 1b) Select platform
    platform = mm.Platform.getPlatformByName(platform_name)

    # 2) Add a torsion restraint force (PeriodicTorsionForce per OpenMM cookbook)
    k_restraint = _as_kjmol(k_restraint)
    restraint = mm.PeriodicTorsionForce()
    system.addForce(restraint)
    # periodicity = 1, phase (theta0) will be updated each step, k in kJ/mol
    torsion_index = restraint.addTorsion(
        dihedral_idx[0], dihedral_idx[1], dihedral_idx[2], dihedral_idx[3],
        1,
        0.0,  # placeholder phase (radians)
        k_restraint.value_in_unit(omm_unit.kilojoule_per_mole)
    )

    # 3) Set up Simulation
    integrator = mm.LangevinIntegrator(
        300 * omm_unit.kelvin,
        1.0 / omm_unit.picosecond,
        2.0 * omm_unit.femtoseconds,
    )
    sim = Simulation(topo, system, integrator, platform)
    sim.context.setPositions(init_pos)

    # ---------- helpers ----------
    def compute_dihedral(p0, p1, p2, p3):
        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

    # 4) Scan loop
    records = []
    with open(xyz_path, "w") as fh:
        for ang in tqdm.tqdm(angles_deg, desc="Dihedral scan"):
            # b) update torsion phase (PeriodicTorsionForce)
            target_rad = math.radians(ang)
            restraint.setTorsionParameters(
                torsion_index,
                dihedral_idx[0], dihedral_idx[1], dihedral_idx[2], dihedral_idx[3],
                1,
                target_rad,
                k_restraint.value_in_unit(omm_unit.kilojoule_per_mole)
            )
            restraint.updateParametersInContext(sim.context)

            # c) minimize *in this same Context* so the torsion restraint remains active
            LocalEnergyMinimizer.minimize(sim.context, minimize_tol, max_iters)

            # e) record energy/coords
            state = sim.context.getState(getEnergy=True, getPositions=True)
            energy_kj = state.getPotentialEnergy().value_in_unit(omm_unit.kilojoule_per_mole)
            positions = state.getPositions()
            pos_nm = positions.value_in_unit(omm_unit.nanometer)
            dihed = compute_dihedral(pos_nm[dihedral_idx[0]], pos_nm[dihedral_idx[1]], pos_nm[dihedral_idx[2]], pos_nm[dihedral_idx[3]])
            write_xyz_frame(fh, topo, positions, comment=f"angle={ang} E={energy_kj:.4f} kJ/mol")
            records.append({"angle": ang, "energy_kJ_mol": energy_kj, "achieved_angle": dihed})
        
    return pd.DataFrame(records)


def scan_dihedral_xtb(
    smi: str,
    dih_smarts: str,
    angles_deg=range(0, 361, 10),
    method: str = "GFN2-xTB",
    charge: int = 0,
    spin: int = 1,
    fmax_ev_per_A: float = 0.05,
    xyz_path: str | Path = "xtb_scan.xyz",
) -> pd.DataFrame:
    """
    Relaxed dihedral scan with ASE + xTB.

    Parameters
    ----------
    smi : str
        SMILES string for the molecule.
    dih_smarts : str
        SMARTS pattern that captures exactly four atoms (the dihedral).
    angles_deg : iterable[int]
        Angles to scan (degrees).
    method : {"GFN2-xTB","GFN1-xTB","GFN0-xTB"}
        xTB method passed to the ASE XTB calculator.
    charge : int
        Total charge of the system.
    spin : int
        Spin multiplicity (1 = singlet).
    fmax_ev_per_A : float
        Convergence criterion for BFGS in eV/Å.
    xyz_path : str | Path
        Output multi-XYZ path.

    Returns
    -------
    pd.DataFrame
        Columns: angle, energy_kcal_mol.
    """
    if XTB is None:
        raise ImportError("xtb.ase.calculator.XTB not available. Install xtb-python/ASE bindings.")

    # Build RDKit mol, add Hs, embed 3D
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()

    # Find dihedral by SMARTS
    pat = Chem.MolFromSmarts(dih_smarts)
    if pat is None:
        raise ValueError(f"Invalid SMARTS: {dih_smarts}")
    matches = mol.GetSubstructMatches(pat)
    if not matches:
        raise RuntimeError(f"No match for SMARTS '{dih_smarts}' in molecule.")
    a, b, c, d = matches[0]
    if len({a, b, c, d}) != 4:
        raise RuntimeError("SMARTS must capture exactly 4 unique atoms.")

    # Prepare file
    xyz_path = Path(xyz_path)
    if xyz_path.exists():
        xyz_path.unlink()

    energies = []
    for theta in tqdm.tqdm(angles_deg, desc="xtb Dihedral scan"):
        # Set dihedral
        rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, float(theta))

        # ASE Atoms
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info.update({"charge": charge, "spin": spin})

        # xTB calculator
        atoms.calc = XTB(method=method, charge=charge, uhf=spin-1)

        # Constrain the dihedral during relaxation
        constraint = FixInternals(dihedrals_deg=[[float(theta), [a, b, c, d]]])
        atoms.set_constraint(constraint)

        # Optimize
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax_ev_per_A)

        # Energy & write
        e_ev = atoms.get_potential_energy()  # eV
        e_kJ = float(e_ev) * 96.485
        comment = f"angle={theta} energy_kJ={e_kJ:.6f}"
        ase_write(xyz_path.as_posix(), atoms, format="xyz", comment=comment, append=True)
        energies.append(e_kJ)

    df = pd.DataFrame({"angle": list(angles_deg), "energy_kJ_mol": energies})
    return df










# Helper for writing xyz frames
def scan_dihedral_psi4(
    smi: str,
    dih_smarts: str,
    angles_deg=range(0, 361, 10),
    method: str = "b3lyp",
    basis: str = "6-311g_d_p_",
    memory: str = "500MB",
    charge: int = 0,
    spin: int = 1,
    fmax_ev_per_A: float = 0.05,
    xyz_path: str | Path = "psi4_scan.xyz",
) -> pd.DataFrame:
    """
    Relaxed dihedral scan using ASE + Psi4.

    Parameters
    ----------
    smi : str
        SMILES string for the molecule.
    dih_smarts : str
        SMARTS pattern that captures exactly four atoms (the dihedral).
    angles_deg : iterable[int]
        Angles to scan (degrees).
    method : str
        Quantum method/DFT functional for Psi4 (e.g. 'b3lyp').
    basis : str
        Basis set string for Psi4 (ASE syntax, e.g. '6-311g_d_p_').
    memory : str
        Psi4 memory string (e.g. '500MB').
    charge : int
        Total charge of the system.
    spin : int
        Spin multiplicity (1 = singlet).
    fmax_ev_per_A : float
        Convergence criterion for BFGS in eV/Å.
    xyz_path : str | Path
        Output multi-XYZ path.

    Returns
    -------
    pd.DataFrame
        Columns: angle, energy_kJ_mol
    """
    if Psi4 is None:
        raise ImportError("ase.calculators.psi4.Psi4 not available. Install Psi4 + ASE Psi4 plugin.")

    # Build RDKit mol, add Hs, embed 3D
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    conf = mol.GetConformer()

    # Find dihedral by SMARTS
    pat = Chem.MolFromSmarts(dih_smarts)
    if pat is None:
        raise ValueError(f"Invalid SMARTS: {dih_smarts}")
    matches = mol.GetSubstructMatches(pat)
    if not matches:
        raise RuntimeError(f"No match for SMARTS '{dih_smarts}' in molecule.")
    a, b, c, d = matches[0]
    if len({a, b, c, d}) != 4:
        raise RuntimeError("SMARTS must capture exactly 4 unique atoms.")

    # Prepare file
    xyz_path = Path(xyz_path)
    if xyz_path.exists():
        xyz_path.unlink()

    energies = []
    for theta in tqdm.tqdm(angles_deg, desc="psi4 Dihedral scan"):
        # Set dihedral
        rdMolTransforms.SetDihedralDeg(conf, a, b, c, d, float(theta))

        # ASE Atoms
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        positions = [tuple(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.info.update({"charge": charge, "spin": spin})

        # Psi4 calculator (ASE style: pass atoms=, assign to atoms.calc)
        calc = Psi4(
            atoms=atoms,
            method=method,
            basis=basis,
            memory=memory,
            num_threads=8,
            charge=charge,
            multiplicity=spin,
            symmetry="c1",
        )
        atoms.calc = calc

        # Constrain the dihedral during relaxation
        constraint = FixInternals(dihedrals_deg=[[float(theta), [a, b, c, d]]])
        atoms.set_constraint(constraint)

        # Optimize
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=fmax_ev_per_A)

        # Energy & write
        e_ev = atoms.get_potential_energy()  # eV
        e_kJ = float(e_ev) * EV_TO_KJ_MOL
        comment = f"angle={theta} energy_kJ={e_kJ:.6f}"
        ase_write(xyz_path.as_posix(), atoms, format="xyz", comment=comment, append=True)
        energies.append(e_kJ)

    df = pd.DataFrame({"angle": list(angles_deg), "energy_kJ_mol": energies})
    return df


# Helper for writing xyz frames
def write_xyz_frame(fh, topology, positions, comment=""):
    """Write a single frame to an open filehandle in XYZ format."""
    natoms = topology.getNumAtoms()
    fh.write(f"{natoms}\n{comment}\n")
    for atom, p in zip(topology.atoms(), positions):
        x, y, z = p.value_in_unit(omm_unit.angstrom)
        fh.write(f"{atom.element.symbol:2s} {x:20.10f} {y:20.10f} {z:20.10f}\n")
