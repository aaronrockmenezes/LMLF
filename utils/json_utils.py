import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import subprocess
import sys

import time
import shutil

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


def write_to_jsonl(molecules: List[dict], database_path: str, mode: str = 'a'):
    # json dump the new molecules into the database opened in mode (eg. 'w', 'a', 'r')
    assert mode == 'a' or mode == 'w' or mode == 'w+', f"Mode must be 'w', 'w+' or 'a'."
    with open(database_path, mode) as f:
        for mol in molecules:
            f.write(json.dumps(mol) + '\n')


def create_new_folder(directory_path: str,
                      custom_suffix: str = None,
                      initial_jsonl_file: str = 'dockingzel.jsonl'):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    # Create a new folder with a custom suffix
    if custom_suffix:
        folder_name = f"run_{custom_suffix}_{current_time}"
    else:
        folder_name = f"run_{current_time}"

    # make new folder in results folder
    folder_path = os.path.join(directory_path, 'results', folder_name)
    if not os.path.exists(folder_path):
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
