# Source: Custom implementation for evaluating trained NCA models

# IMPORTS
import yaml
import torch
import numpy as np
import src.utils.utils as utils
import torch.utils.data as data
import src.datasets.Dataset as Dataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit,StratifiedKFold
from src.models.NCA import MaxNCA, ConvNCA, SimpleNCA
from src.losses.LossFunctions import BCELoss
from src.agents.Agent import Agent
import csv

# CONFIGURATION OF EXPERIMENT
config_path = "config.yaml"
with open(config_path) as file:
    config = yaml.safe_load(file)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DATA LOADING
AML_data_path = "datasets/Matek-19/"

X_AML, y_AML = utils.get_data_AML(AML_data_path, show_distribution=False)
X_AML = np.asarray(X_AML)
y_AML = np.asarray(y_AML)

fold = config["fold"]
skf_AML = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf_AML.get_n_splits(X_AML, y_AML)

for i, (train_index, test_index) in enumerate(skf_AML.split(X_AML, y_AML)):
    if i != fold:
        continue
    X_AML_train, X_AML_test = X_AML[train_index], X_AML[test_index]
    y_AML_train, y_AML_test = y_AML[train_index], y_AML[test_index]

AML_train_dataset = Dataset.WBC_Dataset(X_AML_train, y_AML_train, augment=True, resize=config["resize"], dataset="AML")
AML_val_dataset = Dataset.WBC_Dataset(X_AML_test, y_AML_test, resize=config["resize"], dataset="AML")

if config["balance"]:
    AML_sampler = data.WeightedRandomSampler(weights=utils.get_weights(y_AML_train), num_samples=len(AML_train_dataset), replacement=True)
else:
    sampler = None

AML_train_loader = data.DataLoader(AML_train_dataset, sampler=AML_sampler, batch_size=config["batch_size"])
AML_val_loader = data.DataLoader(AML_val_dataset, batch_size=1)

# MODEL LOADING
if config["model"] == "MaxNCA":
    model = MaxNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])
elif config["model"] == "ConvNCA":
    model = ConvNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])
else:
    model = SimpleNCA(channel_n=config["channel_n"], hidden_size=config["hidden_size"])

model.to(device)
model.load_state_dict(torch.load("models/train_trained_on_AML"))
model.eval()

# INITIALIZE AGENT
agent = Agent(model, config["steps"], config["channel_n"], config["batch_size"])

# TEST ACCURACY FUNCTION
def compute_test_accuracy(model, agent, test_loader, device):
    r"""Computes the test accuracy of the trained model on the validation dataset.
        # Args:
            model: Trained neural cellular automata (NCA) model.
            agent: Agent handling data preparation and inference.
            test_loader: DataLoader containing test samples.
            device: CUDA or CPU device.
        # Returns:
            float: Accuracy of the model on the test set.
    """
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = agent.prepare_data((inputs, targets))
            outputs, _, _ = agent.get_outputs((inputs, targets))

            predictions = torch.sigmoid(outputs)
            predicted_labels = torch.argmax(predictions, dim=1)
            true_labels = torch.argmax(targets, dim=1)

            correct += (predicted_labels == true_labels).sum().item()
            total += targets.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    # Save accuracy to CSV
    with open("test_accuracy_30epochs.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test Set", "Accuracy (%)"])
        writer.writerow(["AML", f"{accuracy * 100:.2f}"])
    return accuracy

# COMPUTE TEST ACCURACY
test_accuracy = compute_test_accuracy(model, agent, AML_val_loader, device)
print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
