import openai
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import subprocess
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
import time
import shutil

openai.api_key = 'ADD API KEY HERE'

directory_path = '/home/datalab/BioLLM/pyLMLF-new'
# directory_path = 'C:/Users/aaron/OneDrive/Desktop/Acads/APPCAIR/pyLMLF-new'


class Tee:

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        for f in self.files:
            f.flush()


# open jsonl file
def open_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Open a jsonl file and return a list of dictionaries
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


# Select a random instance with label=1
def select_instance(D: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Function to select a random instance with label=1 from the dataset/background knowledge
    """
    positive_mols = []
    for mol in D:
        if mol['label'] == '1':
            positive_mols.append(mol)
    instance = np.random.choice(positive_mols)
    if len(positive_mols) >= 1:
        return instance['smiles']
    else:
        raise ValueError("No instance with label=1 found in the dataset")


def assemble_system_prompt() -> str:
    system_prompt = 'You are an AI with expertise in chemistry and molecular design. Your task is to generate novel, chemically plausible molecules with potential applications in areas like drug design or material science. You have access to a large database of known molecules and a powerful AI model that can generate new molecules. You are interested in generating a molecule that is valid and not in any known database and is similar to a given molecule. Answer in only SMILES strings. Do not generate any other type of output or English text. When generating molecules:\n \
    - Ensure chemical validity (bonding, valence, etc.).\n \
    - Tailor molecules to specific properties (e.g., hydrophobicity, polarity) if requested.\n \
    - Be creative and propose original, viable structures.\n \
    Answer in only SMILES strings. Do not generate any other type of output or English text.'

    return system_prompt


# TODO:  Modify this function to include all constraints and background knowledge
def assemble_constraints_prompt(constraints: Dict) -> str:
    constraints_prompt = f'The constraints that the resultant molecule must satisfy are as follows:\n \
        1. The docking score of the molecule with respect to Human Dopamine Betas-Hydroxylase must be {constraints["docking_score"][0]} or lower\n \
        2. The molecular weight of the molecule must be between {constraints["molecular_weight"][0]} and {constraints["molecular_weight"][1]}\n \
        3. The LogP score of the molecule must be between {constraints["logp"][0]} and {constraints["logp"][1]}\n \
        4. The SA Score of the molecule must be between {constraints["sas"][0]} and {constraints["sas"][1]}\n \
        5. The molecule must contain the substructure C1=CC(=O)OC2=CC(=C(C=C21)O)O\n'

    return constraints_prompt


def assemble_background_knowledge_prompt(
        background_knowledge: List[Dict]) -> str:
    background_knowledge_prompt = f'You have access to a large database of known molecules labelled 1 for acceptable and 0 for unacceptable. The database contains the following molecules:'
    for i, mol in enumerate(background_knowledge):
        background_knowledge_prompt += f'\nMolecule {i+1}: {mol["smiles"]}'
        if 'label' in mol:
            background_knowledge_prompt += f', Label: {mol["label"]}'
        if 'docking_score' in mol:
            background_knowledge_prompt += f', Docking Score: {mol["docking_score"]:.5f}'
        if 'mw' in mol:
            background_knowledge_prompt += f', Molecular Weight: {mol["mw"]:.5f}'
        if 'logp' in mol:
            background_knowledge_prompt += f', LogP: {mol["logp"]:.5f}'
        if 'sas' in mol:
            background_knowledge_prompt += f', SA Score: {mol["sas"]:.5f}'
        if 'substructure_match' in mol:
            background_knowledge_prompt += f', Contains C1=CC(=O)OC2=CC(=C(C=C21)O)O: {mol["substructure_match"]}.'

    return background_knowledge_prompt


def assemble_query(smiles: str,
                   substructure: str = "C1=CC(=O)OC2=CC(=C(C=C21)O)O",
                   background_knowledge: List[Dict] = None,
                   constraints: Dict = None) -> str:
    query = f'Generating only SMILES strings, generate a novel valid molecule for inhibiting DBH, Dopamine Beta Hydroxylase, that is similar to {smiles} and must contain {substructure} substructure and is not in any known database. which statisfies the following criteria: \n \
    1. It should have an estimated binding affinity at least as good as Nepicastat, which has binding affinity of 5.3 and smiles C1CC2=C(C=C(C=C2C[C@H]1N3C(=CNC3=S)CN)F)F.Cl; and\n \
    2. It should be synthesizable in no more than 5 steps; and\n \
    3. The molecules should be contain Esculetin (C1=CC(=O)OC2=CC(=C(C=C21)O)O) or an Esculetin-like substructure;\n \
    4. Synthesis should use reasonably priced starting materials.\n \
    Below are some constraints and background knowledge to guide you:'

    constraints_query = assemble_constraints_prompt(constraints)
    background_knowledge_query = assemble_background_knowledge_prompt(
        background_knowledge)

    if constraints is not None:
        query = f'{query}\n{constraints_query}'
    if background_knowledge is not None:
        query = f'{query}\n{background_knowledge_query}'
    else:
        # query = f'Generating only SMILES strings, generate a novel valid molecule that is similar to {smiles} and must contain {substructure} substructure and is not in any known database.'
        query = f"Generating only SMILES strings, generate a novel valid molecule for inhibiting DBH, Dopamine Beta Hydroxylase, that is similar to {smiles} and must contain {substructure} substructure and is not in any known database. which statisfies the following criteria: \n \
    1. It should have an estimated binding affinity at least as good as Nepicastat, which has binding affinity of 5.3 and smiles C1CC2=C(C=C(C=C2C[C@H]1N3C(=CNC3=S)CN)F)F.Cl; and\n \
    2. It should be synthesizable in no more than 5 steps; and\n \
    3. The molecules should be contain Esculetin (C1=CC(=O)OC2=CC(=C(C=C21)O)O) or an Esculetin-like substructure;\n \
    4. Synthesis should use reasonably priced starting materials."

    return query


def assemble_ranking_system_prompt() -> str:
    system_prompt = 'You are a chemist working on a project to generate novel molecules. You have access to a large database of known molecules and a powerful AI model that can generate new molecules. You are interested in ranking a set of molecules based on a given parameter. Answer in only a comma separated list of indices the ranked molecules. Do not generate any other type of output or English text.'
    return system_prompt


def assemble_ranking_query(background_knowledge: List[Dict],
                           parameter: str) -> str:
    acceptable_mols = []
    for mol in background_knowledge:
        if mol['label'] == '1':
            acceptable_mols.append(mol)
    query = f'Generating only SMILES strings, rank the following molecules in ascending order (lowest to highest) based on the following parameter: {parameter}. Below are the molecules to be ranked:'
    acceptable_mols_query = assemble_background_knowledge_prompt(
        acceptable_mols)
    query = f'{query}\n{acceptable_mols_query}'
    return query


def get_response_from_gpt(system_prompt: str,
                          query: str,
                          model: str = "gpt-4o",
                          temperature: float = 0.5,
                          message: Optional[List[Dict]] = None,
                          max_tokens: int = 60) -> str:
    """
    Function to get a response from the GPT model

    Parameters:
    ----------
    system_prompt: str
        The system prompt
    query: str
        The query to be sent to the model
    model: str
        The model to use for generating the response
    temperature: float
        The temperature parameter for sampling
    message: List[Dict]
        A list of dictionaries containing the messages. If message is provided,
        system_prompt and query are ignored.

    Returns:
    -------
    new_mol: str
        The new molecule generated by the model
    """
    if message is None:
        message = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': query
        }]
    response = openai.chat.completions.create(
        model=model,
        # model = "gpt-4o",
        messages=message,
        max_tokens=max_tokens,
        temperature=temperature,
        n=1,
        stop=None,
        timeout=60)
    new_mol = response.choices[0].message.content
    return new_mol


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
                # temp_score = calculate_docking_score_dummy(smiles)
                temp_score = calculate_docking_score(smiles)
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
    assert len(
        molecular_weight_interval
    ) == 2, "Molecular Weight interval should be a list containing 2 elements: [min, max]"
    assert len(
        logp_score_interval
    ) == 2, "Logp interval should be a list containing 2 elements: [min, max]"
    assert len(
        sa_score_interval
    ) == 2, "SA Score interval should be a list containing 2 elements: [min, max]"
    assert isinstance(substructure_match,
                      bool), "Substrcuture match should be a bool: True/False"

    if properties is not None:
        print(
            f"Docking Score: {properties['docking_score']:.5f}, Molecular weight: {properties['mw']:.5f}, logP: {properties['logp']:.5f}, SAS: {properties['sas']:.5f}, Substructure match: {properties['substructure_match']}"
        )
    else:
        print("No properties found for the molecule")
        return False

    if (docking_score_interval[0] <= properties['docking_score'] <=
            docking_score_interval[1] and molecular_weight_interval[0] <=
            properties['mw'] <= molecular_weight_interval[1]
            and logp_score_interval[0] <= properties['logp'] <=
            logp_score_interval[1] and
            sa_score_interval[0] <= properties['sas'] <= sa_score_interval[1]
            and properties['substructure_match'] == substructure_match):
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
                f"Molecular Weight: {properties['mw']} not withing threshold")
        if not (logp_score_interval[0] <= properties['logp'] <=
                logp_score_interval[1]):
            print(f"Logp: {properties['logp']} not withing threshold")
        if not (sa_score_interval[0] <= properties['sas'] <=
                sa_score_interval[1]):
            print(f"SA Score: {properties['mw']} not withing threshold")
        if not (properties['substructure_match'] == substructure_match):
            print(f"Substructure match not found")

        return False


def write_to_jsonl(molecules: List[dict], database_path: str, mode: str = 'a'):
    # json dump the new molecules into the database opened in mode (eg. 'w', 'a', 'r')
    assert mode == 'a' or mode == 'w' or mode == 'w+', f"Mode must be 'w', 'w+' or 'a'."
    with open(database_path, mode) as f:
        for mol in molecules:
            f.write(json.dumps(mol) + '\n')


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


# TODO Refine this function
def calculate_docking_score(smiles,
                            directory_path='/home/datalab/BioLLM/aaron_pyLMLF'
                            ):
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
    pdb_filename = '/home/datalab/BioLLM/aaron_pyLMLF/ligand.pdb'
    writer = Chem.PDBWriter(pdb_filename)
    writer.write(molecule)
    writer.close()

    # Run the gnina docking script
    cmd = [
        '/home/datalab/gnina', '--config', '4zel_config.txt', '--ligand',
        '/home/datalab/BioLLM/aaron_pyLMLF/ligand.pdb', '--out', 'output.sdf',
        '--log', '/home/datalab/BioLLM/aaron_pyLMLF/threshold_output_log.txt',
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


def create_new_folder(custom_suffix: str = None,
                      initial_jsonl_file: str = 'dockingzel.jsonl'):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Create a new folder with a custom suffix
    if custom_suffix:
        folder_name = f"run_{custom_suffix}_{current_time}"
    else:
        folder_name = f"run_{current_time}"

    # make new folder in results folder
    folder_path = os.path.join(directory_path, 'results', folder_name)
    os.makedirs(folder_path)

    # Copy initial_jsonl_file to the new folder as dockingzel4.jsonl
    src = os.path.join(directory_path, initial_jsonl_file)
    target_dst = os.path.join(folder_path, initial_jsonl_file)
    shutil.copy(src, target_dst)
    if custom_suffix:
        generated_mols_path = os.path.join(
            folder_path,
            f'generated_mols_{custom_suffix}_{current_time}.jsonl')
    else:
        generated_mols_path = os.path.join(
            folder_path, f'generated_mols_{current_time}.jsonl')
    return target_dst, generated_mols_path, folder_path


def get_parameter_ranking(
        background_knowledge: List[Dict],
        parameter: str,
        model: str = "gpt-4o",
        temperature: float = 0.5,
        quantile_range: List[int] = [0.25, 0.75]) -> List[Dict]:
    """
    Function to rank the molecules based on a given parameter
    
    Parameters:
    ----------
    background_knowledge: List[Dict]
        List of dictionaries containing the background knowledge
    parameter: str
        The parameter based on which the molecules are to be ranked
    model: str
        The model to use for generating the response
    temperature: float
        The temperature parameter for sampling
    quantile: float
        The quantile to be used for ranking the molecules
    
    Returns:
    -------
    updated_background_knowledge: List[Dict]
        List of dictionaries containing the updated background knowledge
    """

    # Get the query
    system_prompt = assemble_ranking_system_prompt()
    query = assemble_ranking_query(background_knowledge, parameter)

    response = get_response_from_gpt(system_prompt=system_prompt,
                                     query=query,
                                     model=model,
                                     temperature=temperature)

    # Split the response into a list of SMILES strings
    ranked_mols_indices = response.split(',')

    acceptable_mols = []
    rejected_mols = []
    for mol in background_knowledge:
        if mol['label'] == '1':
            acceptable_mols.append(mol)
        else:
            rejected_mols.append(mol)

    # Get molecules within given quantile
    ranked_mols = []
    for idx in ranked_mols_indices:
        idx = int(idx)
        ranked_mols.append(acceptable_mols[idx - 1])

    # Update the labels of the molecules
    relablled_ranked_mols = []
    lower_quantile = int(len(ranked_mols) * quantile_range[0])
    upper_quantile = int(len(ranked_mols) * quantile_range[1])
    for i, mol in enumerate(ranked_mols):
        if lower_quantile <= i <= upper_quantile:
            mol.update({'label': '1'})
        else:
            mol.update({'label': '0'})
        relablled_ranked_mols.append(mol)

    # Reassemble the background knowledge by combining the acceptable and rejected mols
    print(len(relablled_ranked_mols), len(rejected_mols),
          len(background_knowledge))

    assert (len(ranked_mols) + len(rejected_mols)) == len(
        background_knowledge
    ), "Sum of length of new set of mols and background knowledge should be the same"

    relablled_ranked_mols.extend(rejected_mols)
    updated_background_knowledge = relablled_ranked_mols

    assert len(updated_background_knowledge) == len(
        background_knowledge
    ), "Length of updated background knowledge should be the same as the original background knowledge"

    return updated_background_knowledge
