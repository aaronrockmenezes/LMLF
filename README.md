# LMLF - Language Model with Logical Feedback

This folder contains 1 utils folder, 1 jsonl file and 1 Python file for the LMLF procedure.

### utils

This folder contains 3 utility files for the LMLF procedure.

- `json_utils.py`: Contains utility functions for reading and writing JSON files.
- `LLM.py`: Contains utility functions for the LLM queries and prompts.
- `properties.py`: Contains utility functions for the molecules' properties. I have included a dummy function with a RNG to fetch docking scores in case users don't have Gnina installed.

### inhibitors.jsonl

This file contains the material properties in JSONL format. Each line is a JSON object with the following keys:
- `composition`: The composition of the material.
- `dielectric_constant`: The dielectric constant of the material.
- `label`: The label of the material.

### lmlf.py

How to run the code?
- Change the directory to the current folder.
- Add in your OpenAI API Key in the `lmlf.py` file. (Line 8)
- Add absolute path of the `inhibitors.jsonl` file in the `lmlf.py` file. (Line 9)
- Enter whatever constraints you wish to optimize the molecules for. (Line 28)
- Enter initial constraints. (Line 31)
- Run the following command: 
```bash
python3 lmlf.py
```

This file contains the main LMLF code. It uses the utils to create a feedback loop between the LLM and the material properties.

NOTE: The results will not be updated in the main `inhibitors.jsonl` file. A new folder is created with each run of the LMLF procedure. The folder is named `inhibitors_<timestamp>` and contains the updated `inhibitors.jsonl` file.

The LMLF procedure is as follows:

1. Initialize constraints and background knowledge. This includes the molecular properties that we want to optimize and some known molecules + properties.
2. Load up molecules' properties from `inhibitors.jsonl`.
3. Initialize the LLM.
4. Randomly sample a molecule with label=1 from `inhibitors.jsonl` and query the LLM to generate a similar but novel molecule which satisfies the constraints.
5. Get the properties of the generated molecule using utils.
6. Update the molecules' properties in `inhibitors.jsonl` with the generated molecules.
7. Update constraints, background knowledge and revalidate the database, i.e. `inhibitors.jsonl`.
8. Repeat steps 4-7 for a number of iterations.