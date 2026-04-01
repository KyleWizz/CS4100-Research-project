from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

#RUN IN TERMINAL
""" 
dataset at 2.19 version for now
Input: a genomic nucleotide sequence centered on the SNP with the 
reference allele at the SNP location, 
a genomic nucleotide sequence centered on the 
SNP with the alternative allele at the SNP location, and tissue type
Output: a binary value referring to whether the variant has a causal effect on gene expression
run in the venv (working on pycharm implementation

DISCLAIMER: env.
torch==2.10.0
datasets==2.19.0 - huggingface dataset wouldnt work with newer versions, pain!
python=3.12 environment since errors w/ databases
pyarrow==15.0.0 - reverted due to other reversions
numpy==1.26.4 - i think any version of numpy works it was giving errors though
- numpy just for evals
scipy==1.11.4
scikit-learn==1.3.2
"""
sequence_length=131072 # (sequence_length // 128) % 2 == 0

#task_name = "variant_effect_causal_eqtl"
#task_name = "bulk_rna_expression"
#vector of 218 different ti ssue types - sequence outputs the same vector matrix [218]
bulk_rna_expression = 218
task_name = "cage_prediction"
# One of:
# ["variant_effect_causal_eqtl","variant_effect_pathogenic_clinvar",
# "variant_effect_pathogenic_omim","cage_prediction", "bulk_rna_expression",
# "chromatin_features_histone_marks","chromatin_features_dna_accessibility",
# "regulatory_element_promoter","regulatory_element_enhancer"]

#goal is to do bulk rna expression eventually, build a CNN baseline -> RNN baseline -> FNO baseline and compare.
#features to have after these three steps : UI and graphs, research paper attributed to this repo.
#currently working on cage prediction as it may be quicker, can continue with bulk later on.

#dataset = load_dataset(
    #"InstaDeepAI/genomics-long-range-benchmark", "bulk_rna_expression",
#dataset load was broken so messed with it - data set 2.19, pyfaidx downloaded, os path changed
os.makedirs("C:/genomics/downloads", exist_ok=True)
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=task_name,
    sequence_length=sequence_length,
    trust_remote_code=True,
    cache_dir="C:/genomics"
    #b_rna_ex outputs 218 - might have to change to work - still WIP
    #seq. wise regression
    #bulk_rna_expression = bulk_rna_expression,
    #subset = True, if applicable - for now instead of training on millions, subset for computational time-sake
)


'''
Generating train split: 33891 examples [03:02, 186.17 examples/s]
Generating validation split: 2195 examples [00:08, 270.32 examples/s]
Generating test split: 1922 examples [00:09, 201.17 examples/s]
Features : 

Dataset({
    features: ['sequence', 'labels', 'chromosome', 'labels_start', 'labels_stop'],
    num_rows: 33891
})
Dataset({
    features: ['sequence', 'labels', 'chromosome', 'labels_start', 'labels_stop'],
    num_rows: 1922
})
'''
training_set = dataset["train"]
test_set = dataset["test"]
epochs = 100
batch_size = 32

print(training_set)

print(test_set)
run_name = f"{datetime.now().strftime('cnn_run-%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")

#one-hot encode
def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq])
    one_hot = torch.zeros(4, len(seq))  # [4, 2048]
    one_hot[indices, torch.arange(len(seq))] = 1
    return one_hot
#debugging
all_labels_list = []
for i in range(min(1000, len(training_set))):  # Sample 1000 examples
    all_labels_list.extend(np.array(training_set[i]['labels']).flatten())

label_mean = np.mean(all_labels_list)
label_std = np.std(all_labels_list)
print(f"Label mean: {label_mean:.4f}")
print(f"Label std: {label_std:.4f}")
print(f"Label max: {np.max(all_labels_list):.4f}")
print(f"Label min: {np.min(all_labels_list):.4f}\n")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.conv2_drop = nn.Dropout(.3)
        #input size matching about 32,384, 256 output size
        self.fc1 = nn.Linear(64 * 509, 256)
        self.fc2 = nn.Linear(256, 800)






    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.conv2_drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x.view(-1, 16, 50) #batch, 16, 50

#need something w/ weighted since model keeps getting stuck (stagnated)
class WeightedHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[target > 0.69] = 5.0
        weights[target > 1.39] = 12.0
        weights[target > 2.40] = 25.0

        # Huber loss with weights
        diff = pred - target
        abs_diff = torch.abs(diff)

        loss = torch.where(
            abs_diff < 1.0,
            0.5 * diff ** 2 * weights,
            (abs_diff - 0.5) * weights
        )
        return loss.mean()


#pip install numpy==1.26.4 for eval
def cnnModel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    # loss = nn.MSELoss()
    loss = WeightedHuberLoss()

    training_loader = torch.utils.data.DataLoader(training_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=0)
    #print(iter(training_loader))
    print(f"Starting training for {epochs} epochs...\n")
    val_set = dataset['validation']
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(training_loader):
            #prep
            sequences = torch.stack([encoding(seq) for seq in batch["sequence"]]).to(device)
            #sequences = sequences.unsqueeze(1).to(device)
            labels_np = np.array(batch["labels"])
            #needed to debug with help
            if labels_np.shape != (batch_size, 16, 50):
                labels_np = labels_np.transpose(2, 0, 1)  # Reorder to [32, 16, 50]
            #normalizing was only way I could decrease the MSE we got during epoch training
            labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
            labels = torch.log1p(labels) #changed to logs from mean-std
            optimizer.zero_grad()
            outputs = model(sequences)
            #add outputs with labels to our loss
            batch_loss = loss(outputs, labels)
            if torch.isnan(batch_loss) or torch.isinf(batch_loss) or batch_loss.item() > 1.5:
                optimizer.zero_grad()
                continue

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred_std = outputs.std().item()
                    pred_mean = outputs.mean().item()
                print(f"Epoch {epoch + 1}/{epochs} | Batch "
                      f"{batch_idx}/{len(training_loader)} "
                      f"| Batch Loss: {batch_loss.item():.4f} "
                      f"| Pred std: {pred_std:.4f}")  # ADD THIS
                writer.add_scalar("Loss/batch", batch_loss.item(), epoch * len(training_loader) + batch_idx)
                writer.add_scalar("Predictions/std", pred_std, epoch * len(training_loader) + batch_idx)
                if pred_std < 0.01:  # ADD THIS CHECK
                    print("Model collapsing! Predictions too constant")

        avg_loss = epoch_loss / len(training_loader)


        #VALIDATION CHECK --

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                sequences = torch.stack([encoding(s) for s in batch["sequence"]]).to(device)
                labels_np = np.array(batch["labels"])
                if labels_np.shape != (labels_np.shape[0], 16, 50):
                    labels_np = labels_np.transpose(2, 0, 1)
                labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
                labels = torch.log1p(labels)
                outputs = model(sequences).view(-1, 16, 50)
                val_loss += loss(outputs, labels).item()
        val_loss /= len(val_loader)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        best_val_loss = float('inf')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "CNN_testline/cnn_best_checkpoint.pth")
            print(f"  New best val: {best_val_loss:.4f}")
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}/{epochs} | avg: {avg_loss:.4f} | val: {val_loss:.4f}")

        print(f"Epoch {epoch+1}/{epochs}.. " + f" | {avg_loss}")
        #--
    writer.close()
    os.makedirs("CNN_testline", exist_ok=True)
    torch.save(model.state_dict(), "CNN_testline/cnn_baseline2.pth")
    print("Training complete! saved to cnn_baseline2.pth")

    return model

if __name__ == '__main__':
    cnnModel()