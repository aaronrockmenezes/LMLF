import openai
import numpy as np
import pandas as pd
import json
import os
from typing import Optional, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Contrib.SA_Score import sascorer
import time
from utils.utils import *

openai.api_key = 'ADD API KEY HERE'

# Notation
"""
L: LLM
B: Backgrond Knowledge
Q: Query
C: Constraints
repsonse: LLM response
k: Max iterations
i: Current iteration
n: Max samples per iteration
j: Current sample
"""

# k = Max iterations
k = 20
# n = Max samples per iteration
n = 5
# u = constraint update frequency
u = 5
# r = Parameters to optimize
r = ['docking_score', 'molecular_weight', 'sa_score', 'logp']


# TODO: Make a function to update constraints based on r
# make dummy function
def update_constraints(constraints, phase):
    if phase == 0:
        constraints['docking_score'][0] += 0.5
    elif phase == 1:
        constraints['molecular_weight'][1] -= 50
    elif phase == 2:
        constraints['sas'][1] -= 0.25
    elif phase == 3:
        constraints['logp'][1] -= 0.25
    return constraints


target_mols_file, generated_mols_file, folder_path = create_new_folder(
    custom_suffix='gpt40_mini_1', initial_jsonl_file="dockingzel.jsonl")

# Open a file to write the output
log_file = open(os.path.join(folder_path, 'output_logs.txt'), 'w')

# Redirect stdout to both the terminal and the file
sys.stdout = Tee(sys.stdout, log_file)

# Initial Constraints, keep them low
constraints = {}
constraints['docking_score'] = [4.5, np.inf]
constraints['molecular_weight'] = [200, 700]
constraints['logp'] = [0, 5]
constraints['sas'] = [0, 5]
constraints['substructure_match'] = True

generated_mols = set()

for phase in range(len(r)):
    print(f"------------ Phase {phase} - Optimizing {r[phase]} ------------")
    # Vary i from 1 to k
    for i in range(1, k + 1):

        # Keep track of mols in db to avoid duplicates
        D = open_jsonl(target_mols_file)
        mols_in_db = {mol['smiles'] for mol in D}

        print(f"------------ Iteration {i} ------------")
        j = 1
        E = set()
        print(f"Current constraints:-")
        for key, items in constraints.items():
            print(f"{key}: {items}")

        # Vary j from 1 to n
        for j in range(1, n + 1):
            print(f"------------ Sample {j} ------------")
            sample = select_instance(D)
            print(f"Sampled molecule: {sample}")
            system_prompt = assemble_system_prompt()
            query = assemble_query(sample,
                                   substructure="C1=CC(=O)OC2=CC(=C(C=C21)O)O",
                                   background_knowledge=D,
                                   constraints=constraints)
            response = get_response_from_gpt(system_prompt=system_prompt,
                                             query=query,
                                             model="gpt-4o-mini-2024-07-18",
                                             temperature=0.7)
            validity = check_validity(response)
            if validity and (response not in E) and (
                    response not in mols_in_db) and (response
                                                     not in generated_mols):
                print(f"New molecule generated: {response}\n")
                E.add(response)
                generated_mols.add(response)
            elif validity:
                generated_mols.add(response)
                print('Duplicate molecule generated. Skippping...\n')
            else:
                print('Invalid molecule generated. Skippping...\n')
        print(f"Total unique molecules generated: {len(E)}")

        # Check if new batch satisfies constraints
        acceptable_mols = []
        rejected_mols = []
        for smiles in E:
            properties = get_properties(
                smiles, substructure="C1=CC(=O)OC2=CC(=C(C=C21)O)O")
            properties.update({'generation': i})
            properties.update({'phase': phase})
            if check_constraints(smiles, properties, constraints):
                print(f"New molecule satisfies constraints: {smiles}\n")
                properties.update({'label': '1'})
                acceptable_mols.append(properties)
            else:
                print(f"New molecule does not satisfy constraints: {smiles}\n")
                properties.update({'label': '0'})
                rejected_mols.append(properties)

        print(f"Total molecules accepted: {len(acceptable_mols)}/{len(E)}")
        print("Writing to file ./generated_mols_with_properties.jsonl")
        write_to_jsonl(acceptable_mols, generated_mols_file)
        write_to_jsonl(rejected_mols, generated_mols_file)

        print(f"Updating database...")
        write_to_jsonl(acceptable_mols, target_mols_file)
        print(f"Database updated.\n")

        # Re-validate database and change constraints every 5 iterations
        if (i % u == 0):
            D = open_jsonl(target_mols_file)
            if i != k:
                constraints = update_constraints(constraints, phase)

            print(f"--------- Validating Database ---------")
            # Check if properties already present, if so use them and don't recalc
            new_mols = validate_db(database=D, constraints=constraints)

            # if phase == 1:
            #     new_mols = get_parameter_ranking(
            #         background_knowledge=new_mols,
            #         parameter=r[phase],
            #         model='gpt-4o-mini-2024-07-18',
            #         temperature=0.7,
            #         quantile_range=[0, 0.25])
            # elif phase == 2:
            #     new_mols = get_parameter_ranking(
            #         background_knowledge=new_mols,
            #         parameter=r[phase],
            #         model='gpt-4o-mini-2024-07-18',
            #         temperature=0.7,
            #         quantile_range=[0.75, 1.0])

            write_to_jsonl(new_mols, target_mols_file, mode='w')
            print("------------ Database Validated ------------\n")
    if phase != len(r) - 1:
        print(f"------------ Moving to phase {phase+1} ------------\n")
    elif phase == len(r) - 1:
        print(f"------------ Optimization Complete ------------\n")

print(f"------------ Terminating LMLF ------------\n")
