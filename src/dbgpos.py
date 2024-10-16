import numpy as np
import networkx as nx 
import torch
from torch_geometric.utils import from_networkx
import torch
import matplotlib.pyplot as plt
from torch_geometric.nn.encoding import PositionalEncoding


aminoacidi = 'ACDEFGHIKLMNPQRSTVWYX'

def one_hot_encoding(sequence):
    """
    This function performs one-hot encoding on a given protein sequence.

    Parameters:
    sequence (str): The protein sequence to be encoded. The sequence should only contain amino acids.

    Returns:
    torch.Tensor: A one-hot encoded tensor representing the input protein sequence. The tensor has shape (len(aminoacidi), len(sequence)).
    """
    aa_to_index = {aa: i for i, aa in enumerate(aminoacidi)}
    one_hot = torch.zeros((len(sequence), len(aminoacidi)))
    for i, aa in enumerate(sequence):
        one_hot[i, aa_to_index[aa]] = 1

    one_hot=one_hot.T.flatten()
    return one_hot

def generate_all_kmers(sequence,k):
    kmers=[sequence[i:i+k] for i in range(len(sequence)-k+1)]
    return kmers
def generate_all_kmers(sequence, k):
    """
    This function generates all possible k-mers from a given protein sequence.

    Parameters:
    sequence (str): The protein sequence from which to generate k-mers. The sequence should only contain amino acids.
    k (int): The length of the k-mers to be generated.

    Returns:
    list: A list of all possible k-mers extracted from the input protein sequence.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
    return kmers



def create_positional_encoding_kmers(kmers, d):
    """
    This function generates positional encodings for all k-mers in a given protein sequence and stores them in a dictionary.

    Parameters:
    kmers (list): A list of all possible k-mers extracted from the input protein sequence.
    d (int): The dimension of the positional encoding.

    Returns:
    dict: A dictionary where the keys are the k-mers and the values are their corresponding positional encodings.
    """
    kmers_pos_enc = {}
    kmers_count = {}
    positional_encoding = PositionalEncoding(d)

    for i, kmer in enumerate(kmers):
        pe = positional_encoding(torch.tensor([[i]])).squeeze(0).tolist()

        if kmer in kmers_pos_enc:
            kmers_pos_enc[kmer] = [sum(x) for x in zip(kmers_pos_enc[kmer], pe)]
            kmers_count[kmer] += 1
        else:
            kmers_pos_enc[kmer] = pe
            kmers_count[kmer] = 1

    for kmer in kmers_pos_enc:
        kmers_pos_enc[kmer] = [x / kmers_count[kmer] for x in kmers_pos_enc[kmer]]

    return kmers_pos_enc

def create_dbg(sequence, k, emb_size):
    """
    This function creates a directed bipartite graph (DBG) from a given protein sequence.

    Parameters:
    sequence (str): The input protein sequence from which to generate the DBG. The sequence should only contain amino acids.
    k (int): The length of the k-mers to be generated.
    emb_size (int): The dimension of the embedding for the nodes in the DBG.

    Returns:
    torch_geometric.data.Data: A PyTorch Geometric Data object representing the created directed bipartite graph. The data object contains the graph structure (G.x) and the corresponding node features (G.x).

    The function first generates all possible k-mers from the input protein sequence. Then, it creates a directed bipartite graph where each node represents a k-mer, and an edge connects a prefix and its corresponding suffix. The function then computes one-hot encoding and positional encoding for each node, concatenates them, and assigns them as node features in the graph. Finally, it returns the created directed bipartite graph as a PyTorch Geometric Data object.
    """
    k_1_mers = generate_all_kmers(sequence, k - 1)
    d = emb_size - (len(aminoacidi) * (k - 1))
    pos_enc = create_positional_encoding_kmers(k_1_mers, d)
    g = nx.Graph()
    feats = []
    for kmer in generate_all_kmers(sequence, k):
        prefix = kmer[:-1]
        suffix = kmer[1:]
        g.add_edge(prefix, suffix)
    for node in g.nodes():
        ohe = one_hot_encoding(node)
        pe = torch.tensor(pos_enc[node])
        feats.append(torch.concatenate([ohe, pe]))

    feats = torch.stack(feats)

    G = from_networkx(g)
    G.x = feats
    return G


sequence="MASPREENVYLAKLAEQAERYEEMVEFMEKVVGAGDDELTIEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNEDHVASIKTYRSKIESELTSICNGILKLLDSKLIGTAATGDSKVFYLKMKGDYYRYLAEFKTGAERKEAAENTLSAYKSAQDIANGELAPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAIAELGTLGEESYKDSTLIMQLFCDNLTLWTSDMQDDGTDEIKEPSKAEEQQ"
k=5
emb_size=128
g=create_dbg(sequence,k,emb_size)

print(g)

