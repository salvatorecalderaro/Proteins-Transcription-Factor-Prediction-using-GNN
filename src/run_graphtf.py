import random
import os 
import numpy as np 
import torch 
import platform
import cpuinfo
import argparse
from Bio import SeqIO
from dbgpos import create_dbg
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from GraphTFactor import GraphTFactor, train_net,predict
from evaluation import evaluate_model,compute_roc_curve,plot_mean_roc_curve,plot_mean_cm,save_results

seed=0
epochs=100
nfolds=10
mini_batch_size=128
lr=0.001
hidden_channels=512


def set_seed(seed):
    """
    This function sets the random seed for various libraries used in the script.

    Parameters:
    - seed (int): The random seed to be used.

    The function sets the random seed for the 'random' library, the environment variable 'PYTHONHASHSEED', the 'numpy' library, the 'torch' library, and the 'cudnn' backend of 'torch'. This ensures that the results of the experiments are reproducible.

    Returns:
    None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)

def identify_device():
    """
    This function identifies the device to be used for training and testing the model.

    Parameters:
    - None

    Returns:
    - device (torch.device): The device to be used for training and testing the model. Can be "cpu" or "cuda" if available.
    - dev_name (str): The name of the device.

    The function first determines the operating system of the machine. If the operating system is Darwin (macOS), it checks if the MPS (Metal Performance Shaders) backend is available. If so, it sets the device to "mps"; otherwise, it sets the device to "cpu". If the operating system is not Darwin, it checks if the CUDA backend is available. If so, it sets the device to "cuda"; otherwise, it sets the device to "cpu". The function then retrieves the name of the device and returns both the device and its name.
    """
    so = platform.system()
    if (so == "Darwin"):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        d = str(device)
        if d == 'cuda':
            dev_name = torch.cuda.get_device_name()
            set_seed(seed)
        else:
            dev_name = cpuinfo.get_cpu_info()["brand_raw"]
    return device, dev_name

def parse_arguments():
    """
    This function parses the command-line arguments for the script.

    Parameters:
    - parser (argparse.ArgumentParser): The ArgumentParser object used to define the command-line arguments.

    Returns:
    - Tuple[str, int, int]: A tuple containing the dataset name, k-Mers size, and embedding size.

    The function uses the argparse module to define and parse the command-line arguments. It specifies that the user must provide three arguments: "-d" (dataset name), "-k" (k-Mers size), and "-e" (embedding size). The function then returns a tuple containing the parsed values of these arguments.

    The function is designed to be used as a standalone function, but it can also be called within the main function of the script. In this case, the function is called without the 'parser' argument, and the parser object is passed as a global variable.
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("-d", "--dname", type=str, choices=["All", "Eukaryotic", "Prokaryotic", "Virus"], required=True, help="Dataset name")
    parser.add_argument("-k", "--ksize", type=int, required=True, help="k-Mers size")
    parser.add_argument("-e","--embsize",type=int, required=True, help="Embedding size")
    args = parser.parse_args()
    d = args.dname
    k = args.ksize
    es=args.embsize
    return d,k,es

def load_sequences(data_type):
    """
    This function loads sequences and their corresponding labels from a given dataset.

    Parameters:
    - data_type (str): The type of the dataset to use. Can be "All", "Eukaryotic", "Prokaryotic", or "Virus".

    Returns:
    - sequences (list): A list of strings representing the sequences of amino acids.
    - labels (list): A list of integers representing the labels (0 for no-TF and 1 for TF) for each sequence.

    The function reads the sequences and their corresponding labels from a FASTA file corresponding to the specified dataset. It iterates over the input sequences and their labels, and for each sequence, it appends it to the 'sequences' list and its corresponding label to the 'labels' list. The function also prints the number of sequences in the dataset.
    """
    sequences, labels = [], []
    mapping = {"tf": 1, "no-tf": 0}
    path = f"../datasets/{data_type}.fasta"
    with open(path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequence = str(record.seq)
            sequences.append(sequence)
            label = str(record.description).split(" ")[-1]
            labels.append(mapping[label])

    sequences = np.asarray(sequences)
    labels = np.array(labels).reshape(-1)
    print("Number of sequences in the dataset:", sequences.shape[0])
    return sequences, labels

def create_graphs(k, e, sequences, labels):
    """
    This function creates graphs from a list of sequences and their corresponding labels.

    Parameters:
    - k (int): The size of the k-mers to use for creating the graphs.
    - e (int): The embedding size for the nodes in the graphs.
    - sequences (list): A list of strings representing the sequences of amino acids.
    - labels (list): A list of integers representing the labels (0 for no-TF and 1 for TF) for each sequence.

    Returns:
    - graphs (list): A list of DBG objects representing the created graphs.
    - valid_labels (list): A list of integers representing the labels for each graph in the 'graphs' list.

    The function creates a list of DBG objects (graphs) from the input sequences and their corresponding labels. It iterates over the input sequences and their labels, and for each sequence, it creates a DBG object if its length is greater than or equal to 'k'. The DBG object is then added to the 'graphs' list, along with its corresponding label. The function also prints the number of sequences with less than 'k' amino acids.
    """
    graphs = []
    valid_labels = []
    print("Creating DBGs...")
    n = 0

    for (s, l) in tqdm(zip(sequences, labels), total=len(sequences)):
        if len(s) >= k:
            graph = create_dbg(s, k, e)
            graph.y = torch.tensor(l)
            graphs.append(graph)
            valid_labels.append(l)

        else:
            n += 1

    print(f"Number of sequences with less than {k} amino acids: {n}")

    return graphs, valid_labels


def create_train_test_sets(graphs, train, test):
    """
    This function creates training and testing datasets from the given graphs.

    Parameters:
    - graphs (list): A list of DBG objects representing the input graphs.
    - train (list): A list of indices of the graphs to be included in the training set.
    - test (list): A list of indices of the graphs to be included in the test set.

    Returns:
    - trainloader (DataLoader): A DataLoader object containing the training dataset.
    - testloader (DataLoader): A DataLoader object containing the test dataset.

    The function creates two DataLoader objects, one for the training dataset and one for the test dataset. The training dataset is created by selecting the graphs at the indices specified in the 'train' list, while the test dataset is created by selecting the graphs at the indices specified in the 'test' list. The 'batch_size' parameter specifies the number of samples to include in each batch, and the 'shuffle' parameter specifies whether the order of the samples in the dataset should be shuffled.

    The function also prints the number of sequences in the training and test sets.
    """
    data = []
    for i in train:
        data.append(graphs[i])
    print("Number of sequences in the training set:", len(data))
    trainloader = DataLoader(data, batch_size=mini_batch_size, shuffle=True)

    data = []
    for i in test:
        data.append(graphs[i])
    print("Number of sequences in the test set:", len(data))
    testloader = DataLoader(data, batch_size=mini_batch_size, shuffle=False)

    return trainloader, testloader
    data=[]
    for i in train:
        data.append(graphs[i])
    print("Number of sequences in the training set:",len(data))
    trainloader = DataLoader(data, batch_size=mini_batch_size, shuffle=True)

    data=[]
    for i in test:
        data.append(graphs[i])
    print("Number of sequences in the test set:",len(data))
    testloader = DataLoader(data, batch_size=mini_batch_size, shuffle=False)

    return trainloader,testloader

def run_experiment(device, data_type, k, e):
    """
    This function runs an experiment using the GraphTFactor model on a given dataset.

    Parameters:
    - device (str): The device to use for training and testing the model. Can be "cpu" or "cuda" if available.
    - data_type (str): The type of the dataset to use. Can be "All", "Eukaryotic", "Prokaryotic", or "Virus".
    - k (int): The size of the k-mers to use for creating the graphs.
    - e (int): The embedding size for the nodes in the graphs.

    Returns:
    None

    The function does not return any value. It prints the device information,
    dataset details, and experiment results.
    """
    # Load sequences and labels from the dataset
    sequences, labels = load_sequences(data_type)

    # Create graphs and labels for the dataset
    graphs, labels = create_graphs(k, e, sequences, labels)

    # Initialize lists to store the results of the experiment
    cms, aurocs, tprs = [], [], []
    report = []

    # Create a stratified k-fold object for cross-validation
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=seed)

    # Iterate over each fold of the cross-validation
    for f, (train, test) in enumerate(skf.split(graphs, labels), start=1):
        # Print information about the current fold
        print("\n================================================")
        print(f"\nFold {f}/{nfolds}")

        # Create train and test loaders for the current fold
        trainloader, testloader = create_train_test_sets(graphs, train, test)

        # Initialize the GraphTFactor model and train it on the training data
        net = GraphTFactor(e, hidden_channels, num_classes=1).to(device)
        net, t = train_net(device, net, trainloader, epochs, lr)

        # Predict the labels for the test data
        y_true, y_pred, proba = predict(device, net, testloader)

        # Evaluate the model's performance on the test data
        rep, cm = evaluate_model(f, y_true, y_pred, proba)

        # Append the results of the current fold to the respective lists
        report.append(rep)
        cms.append(cm)
        tpr, auc = compute_roc_curve(y_true, proba)
        tprs.append(tpr)
        aurocs.append(auc)

        # Print information about the current fold's results
        print("================================================\n")

    # Save the results of the experiment
    save_results(report, data_type, "graphtf", k, e)

    # Plot the mean confusion matrix for all folds
    plot_mean_cm(cms, data_type, "graphtf", k, e)

    # Plot the mean ROC curve for all folds
    plot_mean_roc_curve(tprs, aurocs, data_type, "graphtf", k, e)

    # Print a message indicating that the experiment has been completed successfully
    print("Experiment completed successfully")



def main():
    """
    This function is the entry point of the script. It initializes the device,
    parses the arguments, and runs the experiment using the GraphTFactor model.

    Parameters:
    None

    Returns:
    None

    The function does not return any value. It prints the device information,
    dataset details, and experiment results.
    """
    device, devname = identify_device()
    data_type,k,es = parse_arguments()
    print("----------------------------------------------------------------")
    print(f"Using {device} - {devname}")
    print(f"Testing GraphTFactor for {data_type} dataset")
    print(f"DBGs with k={k} and node features dim={es}")
    print(f"Epochs: {epochs}")
    print(f"Mini-batch size: {mini_batch_size}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Learning rate: {lr}")
    print("----------------------------------------------------------------\n")
    
    run_experiment(device, data_type, k, es)
    exit(0)
    print(f"Learning rate: {lr}")
    print("----------------------------------------------------------------\n")
    
    run_experiment(device, data_type, k, es)
    exit(0)

if __name__=="__main__":
    main()

