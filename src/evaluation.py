from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score, auc ,roc_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.rcParams["text.usetex"]=True
dpi=1000

def evaluate_model(f, y_true, y_pred, proba):
    """
    Evaluates a machine learning model using various metrics.

    Parameters:
    - f (str): A unique identifier for the model or experiment.
    - y_true (ndarray): The true labels for the data.
    - y_pred (ndarray): The predicted labels for the data.
    - proba (ndarray): The predicted probabilities for the data.

    Returns:
    - list: A list containing the following metrics:
        - f: The unique identifier for the model or experiment.
        - acc: The accuracy of the model.
        - b_acc: The balanced accuracy of the model.
        - pre: The precision of the model.
        - spe: The specificity of the model.
        - sens: The sensitivity of the model.
        - f1: The F1 score of the model.
        - auroc: The area under the ROC curve of the model.
    """
    acc = accuracy_score(y_true, y_pred)
    b_acc = balanced_accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    spe = recall_score(y_true, y_pred, pos_label=0)
    sens = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, proba)
    auroc = auc(fpr, tpr)
    cm=confusion_matrix(y_true, y_pred,labels=[0,1],normalize="true")
    
    print(f"Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {b_acc:.4f}")
    print(f"Precision: {pre:.4f}")
    print(f"Specificity: {spe:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Confusion Matrix:\n {cm}")

    return [f, acc, b_acc, pre, spe, sens, f1, auroc],cm


def compute_roc_curve(y_true, proba):
    """
    Computes the Receiver Operating Characteristic (ROC) curve for a given set of true labels (y_true) and predicted probabilities (proba).

    Parameters:
    - y_true (ndarray): The true labels for the data.
    - proba (ndarray): The predicted probabilities for the data.

    Returns:
    - tuple: A tuple containing the following:
        - tpr_interp (ndarray): The interpolated true positive rate (TPR) values at different false positive rates (FPR).
        - roc_auc (float): The area under the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, proba)
    mean_fpr = np.linspace(0, 1, 100)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    roc_auc = auc(fpr, tpr)
    return tpr_interp, roc_auc

def plot_mean_roc_curve(tprs, aucs, dataset, exp,k=None,e=None):
    """
    This function plots the average Receiver Operating Characteristic (ROC) curve for a given set of true labels (y_true) and predicted probabilities (proba).

    Parameters:
    - tprs (ndarray): The interpolated true positive rate (TPR) values at different false positive rates (FPR).
    - aucs (ndarray): The area under the ROC curve for each fold.
    - dataset (str): The name of the dataset.
    - exp (str): The name of the experiment.

    Returns:
    - None: This function does not return any value. It only plots the average ROC curve and saves it as an image.

    The function first calculates the mean true positive rate (TPR) and the mean area under the ROC curve (AUC) for the given tprs and aucs.
    It then plots the average ROC curve using the mean_fpr and mean_tpr values. 
    The function also calculates the standard deviation of the AUC and adds 
    it to the title of the plot.

    The function also saves the tprs and aucs data to a pickle file for future reference.

    The plot is saved as an image file in the experiments directory with the name "roc_curve.png".
    """
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    if exp=="graphtf":
        description = f"GraphTFactor (k={k},e={e})"
        path = f"../experiments/{exp}/{dataset}/{k} {e}/roc_curve.png"
        folder_path = f"../experiments/{exp}/{dataset}/{k} {e}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    title = f"{dataset} dataset, {description} - Average ROC Curve"

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='red', label=f'Mean ROC (AUC = {mean_auc:.4f} ± {std_auc:.4f})', lw=1)

    tpr_upper = np.minimum(mean_tpr + np.std(tprs, axis=0), 1)
    tpr_lower = np.maximum(mean_tpr - np.std(tprs, axis=0), 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='red', alpha=0.4)
    plt.plot([0, 1], [0, 1], linestyle='--', color='blue', lw=1, label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)

    data = {
        "tprs": tprs,
        "aucors": aucs
    }
    
    
def plot_mean_cm(cms,dataset,exp,k=None,e=None):
    sum_matrix = np.sum(cms, axis=0)
    avg_matrix = sum_matrix / len(cms)
    
    classes=["no-TF","TF"]
    
    if exp=="graphtf":
        description = f"GraphTFactor (k={k},e={e})"
        path = f"../experiments/{exp}/{dataset}/{k} {e}/cm.png"
        folder_path = f"../experiments/{exp}/{dataset}/{k} {e}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    title = f"{dataset} dataset, {description} - Average Confusion Matrix" 
    
    plt.figure(figsize=(6, 6))
    plt.title(title)
    sns.heatmap(avg_matrix, annot=True, fmt=".2%", cmap='Blues',xticklabels=classes,yticklabels=classes, cbar=False)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.clf()
    plt.close()

def save_results(results, dataset, exp,k=None, e=None):
    """
    Saves the evaluation results of a machine learning model to a CSV file and a YAML file.

    Parameters:
    - results (list): A list containing the evaluation results of the model. Each element in the list should be a list containing the following metrics:
        - f: The unique identifier for the model or experiment.
        - acc: The accuracy of the model.
        - b_acc: The balanced accuracy of the model.
        - pre: The precision of the model.
        - spe: The specificity of the model.
        - sens: The sensitivity of the model.
        - f1: The F1 score of the model.
        - auroc: The area under the ROC curve of the model.
    - dataset (str): The name of the dataset.
    - exp (str): The name of the experiment.

    Returns:
    - None: This function does not return any value. It only saves the evaluation results to a CSV file and a YAML file.

    The function first creates a Pandas DataFrame from the input results and saves it to a CSV file with the name "metrics.csv" in the experiments directory. It then calculates the mean and standard deviation of each metric and saves them to a YAML file with the name "results.yaml" in the experiments directory.
    """
    columns = ["Fold", "Accuracy", "Balanced Accuracy", "Precision", "Specificity", "Sensitivity", "F1 Score", "AUROC","Training Time (s)"]
    metrics = pd.DataFrame(results, columns=columns)
    if exp=="graphtf":
        metrics_path = f"../experiments/{exp}/{dataset}/{k} {e}/metrics.csv"
        res_path = f"../experiments/{exp}/{dataset}/{k} {e}/results.yaml"
        folder_path = f"../experiments/{exp}/{dataset}/{k} {e}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    

    
    metrics.to_csv(metrics_path, index=False)

    columns = ["Accuracy", "Balanced Accuracy", "Precision", "Specificity", "Sensitivity", "F1 Score", "AUROC","Training Time (s)"]

    data = {}
    print(f"Average results:")
    for column in metrics.columns[1:]:
        values = metrics[column]
        mu, sigma = np.mean(values), np.std(values)
        print(f"{column}: mean {mu} sd {sigma}")
        data[column] = {
            "Mean": float(mu),
            "Standard Deviation": float(sigma)
        }

    
    with open(res_path, "w") as file:
        yaml.dump(data, file)