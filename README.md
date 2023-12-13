# RecLink
<h3>Revisiting Document-Level Relation Extraction with Context-Guided Knowledge
Graph Link Prediction </h3>

![Sample Image](https://anonymous.4open.science/r/DocRE-CED1/modeldiagram.png)



## Acknowledgments
This repository contains code adapted from the following research papers for the purpose of document-level relation extraction. We extend our gratitude to the authors for generously sharing their clean and valuable code implementations. 

### 1. Discriminative Reasoning for Document-level Relation Extraction

- **Paper**: [Discriminative Reasoning for Document-level Relation Extraction](https://arxiv.org/abs/2106.01562)
- **Authors**: Wang Xu, Kehai Chen, Tiejun Zhao
- **Year**: 2021

Code implementation can be found here: [GitHub - xwjim/DRN](https://github.com/xwjim/DRN/tree/main)

### 2. Modeling Relational Data with Graph Convolutional Networks

- **Paper**: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
- **Authors**: Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling
- **Year**: 2017

Code implementation can be found here: [GitHub - JinheonBaek/RGCN](https://github.com/JinheonBaek/RGCN)

### 3. Multi-Hop Knowledge Graph Reasoning with Reward Shaping

- **Paper**: [Multi-Hop Knowledge Graph Reasoning with Reward Shaping](https://arxiv.org/abs/1808.10568)
- **Authors**: Xi Victoria Lin, Richard Socher, Caiming Xiong
- **Year**: 2018

Code implementation can be found here: [GitHub - kingsaint/InductiveExplainableLinkPrediction](https://github.com/kingsaint/InductiveExplainableLinkPrediction)


<h3>  Environment </h3>

This code is lastly tested with:
- Python 3.7.x
- PyTorch 1.7.x
- CUDA (11.0)
- torch_geometric 1.7.x, with torch_scatter 2.0.6 and torch_sparse 0.6.9

<h3>Libraries</h3>

- numpy (1.19.1)
- matplotlib (3.3.1)
- torch (1.7.1)
- transformers (4.1.1)
- scikit-learn (0.23.2)
- wandb (0.10.12)
- tqdm (4.9.0)


<h3>Directory structure</h3>

In this directory structure, you have a folder named "Reclink" containing a subdirectory "code." Within the "code" directory, there are several files and subdirectories:
- `link prediction`: Files for link prediction module
- `reasoning_path``: files for generating explanations.
- `Context`: Files for creating context.
- `checkpoint/`: Directory to store model checkpoints.
- `logs/`: Directory to store logs related to the code.
- `models/`: Directory to store trained models.
- `config.py`: Configuration file for the code.
- `data.py`: File containing code related to data processing.
- `test.py`: File for testing the code.
- `train.py`: File for training the code.
- `utils.py`: Utility functions used in the code.

## Datasets

This project utilizes the following datasets:

- **DocRED Dataset:** The DocRED dataset can be accessed [here](https://github.com/thunlp/DocRED) place in data/docred directory

- **Re-DocRED Dataset:** The Re-DocRED dataset is available [here](https://github.com/tonytan48/Re-DocRED) place in data/REDocred directory

- **DWIE Dataset:** The DWIE dataset's repository can be found [here](https://github.com/klimzaporojets/DWIE) place in data/DWIE directory

   To convert DWIE dataset into docred style: follow the guideline from this code: https://github.com/rudongyu/LogiRE


   Please make sure to review the terms and conditions of each dataset before use.



# Training

Follow the steps below to start the training process:
1. Train reasoning module: 
 Navigate to the `code` directory using the following command:
   
   cd code
   ./runBERT.sh gpu_id   
   
2. Train link prediction module:
    python3 RGCN/main.py

# Testing

Navigate to the `code` directory using the following command:
  
   cd code
   ./evalBERT.sh gpu_id     

#  Explanation   
Navigate to the `reasoning_path` folder And download the respective dataset models from the link: https://drive.google.com/drive/folders/1j0ArOF9mJDvsYCgLmr022VRb51YxJ9np
Get the explanation using:
For DocRED dataset: 
./experiment-rs.sh configs/DOCRED-rs.sh --inference <gpu-ID> --save_beam_search_paths
For DWIE dataset:
./experiment-rs.sh configs/DWIE-rs.sh --inference <gpu-ID> --save_beam_search_paths

# Context creation
Navigate to context directory:
Select appropriate datasets and files:

1. Create REBEL triples using REBEL.py  

2. Create triples of entity context using entity_context.py (give entity file here)

3. Create triples of context path using context_path.py (provide both entities to calculate path)

4. You can check these files align with dataset using similarity.py and remove noise.
