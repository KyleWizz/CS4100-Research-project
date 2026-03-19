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
#from torch.optim._muon import Muon
#from muon import Muon
#from muon import MuonWithAuxAdam - maybe
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.nn.utils.rnn as rnn_utils
#some libs imported in case
#RUN IN TERMINAL
""" 
(.venv) PS C:  Users \ User\PycharmProjects\PythonProject\CS4100-Research-project> C:
 Users \ User \PycharmProjects\PythonProject\.venv\Scripts\python.exe .\RNN_baseline.py                                                                                 
dataset at 2.19 version for now
Input: a genomic nucleotide sequence centered on the SNP with the 
reference allele at the SNP location, 
a genomic nucleotide sequence centered on the 
SNP with the alternative allele at the SNP location, and tissue type
Output: a binary value referring to whether the variant has a causal effect on gene expression
run in the venv (working on pycharm implementation
/pip.exe install git+https://github.com/KellerJordan/Muon
DISCLAIMER: 
torch==2.10.0
datasets==2.19.0 - huggingface dataset wouldnt work with newer versions, pain! probably better solution but
i used outdated ones instead
python=3.12 environment since errors w/ databases
pyarrow==15.0.0 - reverted due to other reversions
numpy==1.26.4 - i think any version of numpy works it was giving errors though
- numpy just for evals
scipy==1.11.4
scikit-learn==1.3.2
pandas = 2.2.2 for numpy
experimenting w/ muon - can use optim but 2.5.1 doesnt have moon built into pytorch optim
"""
sequence_length=2048
#trained on 2048 bp len sequence (ACTGN)

#task_name = "variant_effect_causal_eqtl"
#task_name = "bulk_rna_expression"
#vector of 218 different tissue types - sequence outputs the same vector matrix [218]
bulk_rna_expression = 218
task_name = "cage_prediction"
run_name = f"{datetime.now().strftime('rnn_run-%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
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
#since it takes a little!
epochs = 50
batch_size = 32

print(training_set)

print(test_set)

#one-hot encode nucleotide sequences
def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq])
    one_hot = torch.zeros(4, len(seq))  # [4, 2048]
    one_hot[indices, torch.arange(len(seq))] = 1
    return one_hot

#data seems to have some outliers, so weight as well
#CAGE signalling has lots of potential noise, read more:
"""
Citations:
Grigoriadis, D., Perdikopanis, N., Georgakilas, G.K. et al. DeepTSS:
 multi-branch convolutional neural network for transcription start 
 site identification from CAGE data. BMC Bioinformatics 23 (Suppl 2),
  395 (2022). https://doi.org/10.1186/s12859-022-04945-y
  
"""
class WeightedHuberLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        weights[target > 0.3] = 30.0
        weights[target > 0.7] = 60.0
        weights[target > 1.3] = 93.0

        # Huber loss with weights - input tensors
        diff = pred - target
        abs_diff = torch.abs(diff)
        #computer loss
        loss = torch.where(
            abs_diff < 1.0,
            0.5 * diff ** 2 * weights,
            (abs_diff - 0.5) * weights
        )
        return loss.mean()

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

"""
Our Bidirectional LSTM 
Forward func for training
"""
class BidLSTM(nn.Module):
    #4 for A C T G
    def __init__(self, input_size = 4, hidden_size = 256, num_layers = 2, num_classes = 800):
        super(BidLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out.mean(dim=1))
        return out

def rnn_model():
    torch.cuda.set_device(0)
    #get gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    #puts model to gpu
    model = BidLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001)

    #if learning stagnates lower rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    loss = WeightedHuberLoss()
    #trying MSE loss first and then huber to see if more accurate than CNN
    #Huber loss may be better because there are definitely outliers in the data
    training_loader = torch.utils.data.DataLoader(training_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0)
    # print(iter(training_loader))
    print(f"Starting training for {epochs} epochs...\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(training_loader):
            # prep
            sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)
            # sequences = sequences.unsqueeze(1).to(device) - not needed
            labels_np = np.array(batch["labels"])
            # needed to debug with help
            if labels_np.shape != (batch_size, 16, 50):
                labels_np = labels_np.transpose(2, 0, 1)  # Reorder to fit RNN
            # normalizing was only way I could decrease the MSE we got during epoch training
            labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
            labels = (labels - label_mean) / label_std
            optimizer.zero_grad()
            outputs = model(sequences).view(-1, 16, 50)
            # add outputs with labels to our loss
            batch_loss = loss(outputs, labels)
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred_std = outputs.std().item()
                    pred_mean = outputs.mean().item()
                global_step = epoch * len(training_loader) + batch_idx
                writer.add_scalar("Loss/batch", batch_loss.item(), global_step)
                writer.add_scalar("Predictions/std", pred_std, global_step)
                writer.add_scalar("Predictions/mean", pred_mean, global_step)
                print(f"Epoch {epoch + 1}/{epochs} | Batch "
                      f"{batch_idx}/{len(training_loader)} "
                      f"| Batch Loss: {batch_loss.item():.4f} "
                      f"| Pred std: {pred_std:.4f}")

                if pred_std < 0.01:
                    print("Model collapsing! Predictions too constant")

        avg_loss = epoch_loss / len(training_loader)
        scheduler.step(avg_loss)
        if not torch.isnan(outputs).any():
            writer.add_histogram("Predictions/distribution", outputs.detach(), epoch)
            writer.add_histogram("Labels/distribution", labels, epoch)
        else:
            print(f"Skipping histogram at epoch {epoch} — NaN outputs detected")
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        if(epoch == 25):
            #used ai for writer just for easier time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            opt_name = type(optimizer).__name__  # e.g. "Adam"
            checkpoint_name = f"rnn_baseline_epoch{epoch}_{timestamp}_{opt_name}"
            writer_mid = SummaryWriter(log_dir=f"runs/{checkpoint_name}")
            torch.save(model.state_dict(), f"RNN_testline/{checkpoint_name}.pth")
            writer_mid.close()

        print(f"Epoch {epoch + 1}/{epochs}.. " + f" | {avg_loss}")
    writer.close()
    torch.save(model.state_dict(), "RNN_testline/rnn_baseline.pth")
    print("Training complete! saved to rnn_baseline.pth")
    return model


if __name__ == '__main__':
    os.makedirs("RNN_testline", exist_ok=True)
    model = rnn_model()
