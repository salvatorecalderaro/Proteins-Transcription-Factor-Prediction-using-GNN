# Proteins-Transcription-Factor-Prediction-using-GNN
This repository provides an innovative deep-learning methodology designed to predict the presence of transcription factors in protein sequences. The model can accurately classify proteins based on their structural and sequential features by representing sequences as graphs and leveraging the power of Graph Neural Networks (GNNs).

## Datasets

- **All dataset**: Contains 3000 TF and 9000 no-TF sequences sampled without considering the organism type.
- **Eukaryotic dataset**: Contains 3000 TF and 9000 no-TF sequences sampled from sequences belonging to eukaryotic organisms.
- **Prokaryotic dataset**: Contains 3000 TF and 3000 no-TF sequences sampled from sequences belonging to prokaryotic organisms.
- **Virus dataset**: Contains 538 TF and 1614 no-TF sequences sampled from sequences belonging to virus organisms.

## Graph Construction
- Identification of the $k$-mers of the sequence $S$ (the number of possible $k$-mers of $S$ is $L-k+1$ where $L = \lvert S \rvert$)
- Assignment of $(k-1)$-mers to the nodes
- Connect one node to another if the $(k-1)$-mer overlaps another

To each node, we assign a feature vector containing the information about the one-hot encoding of the $(k-1)$-mer and its positional information obtained via positional encoding. 


## Installation

To install the required dependencies, use:

```bash
pip install -r requirements.txt
