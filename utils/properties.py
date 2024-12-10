import numpy as np
from typing import List, Dict, Any
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
import time
import shutil


def check_validity(smiles: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False
        print(f"Molecule: {smiles} has a valid SMILES string")
        return True
    except Exception as e:
        print(f"SMILES parsing error {e}. Skipping molecule: {smiles}")
        return False

# TODO: Modify this function to include all constraints
def get_properties(
        smiles: str,
        substructure: str = "C1=CC(=O)OC2=CC(=C(C=C21)O)O") -> Dict[str, Any]:
    """
    Function to calculate the properties of a molecule given its SMILES string

    Parameters:
    ----------
    smiles: str
        The SMILES string of the molecule
    substructure: str
        Substructure to be matched

    Returns:
    -------
    properties: dict
        Dictionary containing the Docking score, Molecular weight, LogP, SAScore, Substructure match, and timestamp of the molecule
    """
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        substructure_mol = Chem.MolFromSmiles(substructure)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        sas = sascorer.calculateScore(mol)

        docking_scores = []
        print("Calculating Docking Score...")
        for _ in range(0, 3):
            try:
                # TODO: Replace with actual function during experimentation
                temp_score = calculate_docking_score_dummy(smiles)
                # temp_score = calculate_docking_score(smiles)
                docking_scores.append(temp_score)
            except Exception as e:
                print(f"Error {e} while calculating docking score")
        try:
            docking_score = np.median(docking_scores)
        except Exception as e:
            docking_score = 0
        properties = {
            "smiles": smiles,
            "docking_score": docking_score,
            "mw": mw,
            "logp": logp,
            "sas": sas,
            "substructure_match": mol.HasSubstructMatch(substructure_mol),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return properties
    except Exception as e:
        print(f"Error calculating properties for {smiles}: {e}")
        return None

# TODO: Make more general
def check_constraints(smiles: str, properties: dict,
                      constraints: dict) -> bool:
    #   docking_score_interval: list = [4.5, np.inf],
    #   molecular_weight_interval: list = [200, 700],
    #   logp_score_interval: list = [0, 5],
    #   sa_score_interval: list = [0, 5],
    #   substructure_match: bool = True) -> bool:
    """
    Function to check if the molecule satisfies the constraints

    Parameters:
    ----------
    smiles: str
        The SMILES string of the molecule
    properties: dict
        Dictionary containing the properties of the molecule
    constraints: dict
        Dictionary containing the constraints
    docking_score_interval: list
        List containing the minimum and maximum docking score
    molecular_weight_interval: list
        List containing the minimum and maximum molecular weight
    logp_score_interval: list
        List containing the minimum and maximum LogP score
    sa_score_interval: list
        List containing the minimum and maximum SAS score
    substructure_match: bool
        Substructure C1=CC(=O)OC2=CC(=C(C=C21)O)O is present or not
    
    Returns:
    -------
    bool

    Notes:
    ------
    Minimum molecular weight is set to 200
    """
    try:
        docking_score_interval = constraints['docking_score']
    except KeyError:
        print(
            "Docking score interval not found. Using default values: [4.5, np.inf]"
        )
        docking_score_interval = [4.5, np.inf]
    try:
        molecular_weight_interval = constraints['molecular_weight']
    except KeyError:
        print(
            "Molecular weight interval not found. Using default values: [200, 700]"
        )
        molecular_weight_interval = [200, 700]
    try:
        logp_score_interval = constraints['logp']
    except KeyError:
        print("LogP interval not found. Using default values: [0, 5]")
        logp_score_interval = [0, 5]
    try:
        sa_score_interval = constraints['sas']
    except KeyError:
        print("SA Score interval not found. Using default values: [0, 5]")
        sa_score_interval = [0, 5]
    try:
        substructure_match = constraints['substructure_match']
    except KeyError:
        print("Substructure match not found. Using default values: True")
        substructure_match = True

    assert len(
        docking_score_interval
    ) == 2, "Docking Score interval should be a list containing 2 elements: [min, max]"

    if properties is not None:
        print(
            f"Docking Score: {properties['docking_score']:.5f}, Molecular weight: {properties['mw']:.5f}"
        )
    else:
        print("No properties found for the molecule")
        return False

    if (docking_score_interval[0] <= properties['docking_score'] <=
            docking_score_interval[1] and molecular_weight_interval[0] <=
            properties['mw'] <= molecular_weight_interval[1]):
        return True
    else:
        if not (docking_score_interval[0] <= properties['docking_score'] <=
                docking_score_interval[1]):
            print(
                f"Docking score: {properties['docking_score']} not within threshold"
            )
        if not (molecular_weight_interval[0] <= properties['mw'] <=
                molecular_weight_interval[1]):
            print(
                f"Molecular Weight: {properties['mw']} not within threshold")

        return False

def calculate_docking_score_dummy(smiles: str) -> float:
    """
    Dummy function to calculate the docking score of a molecule

    Parameters:
    ----------
    smiles: str
        The SMILES string of the molecule
    
    Returns:
    -------
    float
    """
    return np.random.uniform(4, 6.5)

def validate_db(database: List[Dict], constraints: dict) -> bool:
    # docking_score_interval: list = [5.75, np.inf],
    # molecular_weight_interval: list = [200, 700],
    # logp_score_interval: list = [0, 5],
    # sa_score_interval: list = [0, 5],
    # substructure_match: bool = True) -> bool:
    """
    Function to check if the molecule satisfies the constraints

    Parameters:
    ----------
    mol_json: dict
        Dictionary containing the properties of the molecule
    min_docking_score: float
        Minimum docking score
    max_molecular_weight: float
        Maximum molecular weight
    max_logp_score: float
        Maximum LogP score
    max_sa_score: float
        Maximum SAS score
    substructure_match: bool
        Substructure C1=CC(=O)OC2=CC(=C(C=C21)O)O is present or not
    
    Returns:
    -------
    bool

    Notes:
    ------
    Minimum molecular weight is set to 200
    """

    new_mols = []
    # Check if properties already present, if so use them and don't recalc
    for mol in database:
        smiles = mol['smiles']
        validity = check_validity(smiles)
        if validity:
            # Getting the rest of the properties is very quick. If no docking score, get everything
            if 'docking_score' not in mol:
                print(
                    f"No docking score found. Getting molecule's properties..."
                )
                mol = get_properties(smiles)
            constraints_pass = check_constraints(smiles=smiles,
                                                 properties=mol,
                                                 constraints=constraints)
            # docking_score_interval=docking_score_interval,
            # molecular_weight_interval=molecular_weight_interval,
            # logp_score_interval=logp_score_interval,
            # sa_score_interval=sa_score_interval,
            # substructure_match=substructure_match)
            if constraints_pass:
                mol.update({'label': '1'})
                new_mols.append(mol)
                print(f"{smiles} passed validation. Label=1...\n")
            else:
                mol.update({'label': '0'})
                new_mols.append(mol)
                print(f"{smiles} failed validation. Label=0...\n")
        else:
            print(f"{smiles} is not a valid SMILES string. Removing...\n")

    return new_mols

# TODO Refine this function
def calculate_docking_score(smiles:str,
                            directory_path:str
                            ) -> float:
    molecule = Chem.MolFromSmiles(smiles)
    molecule = Chem.AddHs(molecule)
    # Generate 2D coordinates for the molecule
    try:
        AllChem.Compute2DCoords(molecule)
    except Exception as e:
        print(f"Compute2DCoords Error: {e}. Skipping molecule: {smiles}")
        return None

    # Generate a 3D conformation of the molecule using the ETKDG method
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())

    # Optimize the 3D conformation
    try:
        AllChem.MMFFOptimizeMolecule(molecule)
    except Exception as e:
        print(f"MMFFOptimizeMolecule Error: {e}. Skipping molecule: {smiles}")
        return None

    # Generate a PDB file from the molecule
    pdb_filename = f'{directory_path}/ligand.pdb'
    writer = Chem.PDBWriter(pdb_filename)
    writer.write(molecule)
    writer.close()

    # Run the gnina docking script
    cmd = [
        '/home/datalab/gnina', '--config', '4zel_config.txt', '--ligand',
        f'{directory_path}/ligand.pdb', '--out', 'output.sdf',
        '--log', f'{directory_path}/threshold_output_log.txt',
        '--cpu', '4', '--num_modes', '1'
    ]

    # print("Docking Command:", ' '.join(cmd))
    try:
        # Used stdout=subprocess.DEVNULL to suppress the output since we are only interested in the docking score
        subprocess.run(cmd,
                       check=True,
                       stderr=subprocess.PIPE,
                       stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print("Docking process failed:", e)
        print("Error output:", e.stderr)
        return None

    # Iterate over the files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Consider only the text files
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()

                for i, line in enumerate(lines):
                    if 'affinity' in line.lower() and 'cnn' in line.lower():
                        third_next_line_values = lines[i + 3].split()
                        if len(third_next_line_values) >= 4:
                            try:
                                cnn_affinity = float(
                                    third_next_line_values[3].strip())
                                print(
                                    f"Docking Score for {smiles}: {cnn_affinity}\n"
                                )
                                return cnn_affinity
                            except ValueError:
                                pass

    return None