from typing import Optional, List, Dict, Any
import numpy as np

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
    system_prompt = "You are a chemoinformatics expert that can generate new materials, specifically low-k dielectrics. \
    Your task is to generate novel, chemically plausible materials with low dielectric constants. \
    You have access to a large database of known materials and a powerful AI model that can generate new materials. \
    You are interested in generating a molecule that is valid and not in any known database and is similar to a given molecule. \
    Answer in only compositions, eg. CH4, SiO2, etc. Do not generate any other type of output or English text. When generating molecules:\n \
    - Ensure chemical validity (bonding, valence, etc.).\n \
    - Tailor molecules to specific properties (e.g., hydrophobicity, polarity) if requested.\n \
    - Be creative and propose original, viable structures.\n \
    Answer in only compositions. Do not generate any other type of output or English text."

    return system_prompt


# TODO:  Modify this function to include all constraints and background knowledge
def assemble_constraints_prompt(constraints: Dict) -> str:
    constraints_prompt = f"The constraints that the resultant molecule must satisfy are as follows:\n \
        1. The docking score of the molecule with respect to Human Dopamine Betas-Hydroxylase must be {constraints['docking_score'][0]} or lower\n \
        2. The molecular weight of the molecule must be between {constraints['molecular_weight'][0]} and {constraints['molecular_weight'][1]}\n \
        3. The LogP score of the molecule must be between {constraints['logp'][0]} and {constraints['logp'][1]}\n \
        4. The SA Score of the molecule must be between {constraints['sas'][0]} and {constraints['sas'][1]}\n \
        5. The molecule must contain the substructure C1=CC(=O)OC2=CC(=C(C=C21)O)O\n"
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


def get_response_from_gpt(client,
                          system_prompt: str,
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


def get_parameter_ranking(
        client,
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

    response = get_response_from_gpt(client,
                                     system_prompt=system_prompt,
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
