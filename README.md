# Proteins-Transcription-Factor-Prediction-using-GNN

## Overview
This repository provides an innovative deep-learning methodology designed to predict the presence of transcription factors in protein sequences. The model can accurately classify proteins based on their structural and sequential features by representing sequences as graphs and leveraging the power of Graph Neural Networks (GNNs).


## Methodology

1.  Protein sequences are represented as De Bruijn graphs, where the nodes correspond to (k-1)-mers and the edges represent k-mers. An edge is created between two nodes if a (k-1)-mer overlaps another. o initialize the node features, a combination of one-hot encoding and positional encoding is applied. One-hot encoding captures the identity of the (k-1)-mer, while positional encoding introduces the sequence order, allowing the model to learn both structural and positional information about the protein sequence.
