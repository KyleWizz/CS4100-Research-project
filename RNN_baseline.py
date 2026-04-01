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

from scipy.stats import pearsonr, spearmanr #just checking
from scipy.signal import find_peaks
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

epochs = 250
batch_size = 32

print(training_set)

print(test_set)

#one-hot encode nucleotide sequences w/o projection
def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq])
    one_hot = torch.zeros(4, len(seq))  # now [4, 2048]
    one_hot[indices, torch.arange(len(seq))] = 1
    return one_hot

#with proj
# def encoding(seq):
#     mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
#     seq = seq[::4]
#     indices = torch.tensor([mapping.get(base, 0) for base in seq])
#     #projection dimension to lower spaces
#     #512
#     one_hot = torch.zeros(4, len(seq))  # [4, 2048]
#     one_hot[indices, torch.arange(len(seq))] = 1
#     return one_hot

#data seems to have some outliers, so weight as well
#CAGE signaling has lots of potential noise, read more:
"""
Citations:
Grigoriadis, D., Perdikopanis, N., Georgakilas, G.K. et al. DeepTSS:
 multi-branch convolutional neural network for transcription start 
 site identification from CAGE data. BMC Bioinformatics 23 (Suppl 2),
  395 (2022). https://doi.org/10.1186/s12859-022-04945-y
  
"""
class WeightedHuberLoss(nn.Module):
    def __init__(self, label_mean, label_std):
        super().__init__()
        #try this
        # self.t1 = (1.0 - label_mean) / label_std
        # self.t2 = (3.0 - label_mean) / label_std
        # self.t3 = (10.0 - label_mean) / label_std

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        #testing with normalized

        # weights[target > self.t1] = 5.0
        # weights[target > self.t2] = 16
        # weights[target > self.t3] = 32

        #TESTING WEIGHTS - very high - 150x capped at 15 weight loss!!!!!
        #so, "higher batch loss" but it's ok I think
        weights[target > 0.69] = 5.0
        weights[target > 1.39] = 12.0
        weights[target > 2.40] = 25.0
        #PARAMETERIZE - DELTA - adjust weighting - less outlier = MSE more outlier = MAE?
        #ex . delta <= 1.5 mse, or delta > 1.5 for mae
        # weights[target > 0.32] = 30.0
        # weights[target > 0.72] = 60.0
        # weights[target > 1.4] = 95.4 #weights may be aggressive/not since arbitrary

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
    #input size for now: 64 with conv layer
    #test w/ 512 hidden size from 256
    def __init__(self, input_size = 4, hidden_size = 256, num_layers = 2, num_classes = 800):
        super(BidLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.15)
        self.dropout = nn.Dropout(0.27) #signs of overfitting
        #https://codesignal.com/learn/courses -- good resource
        # /improving-neural-networks-with-pytorch/lessons/adding-dropout-to-neural-networks-in-pytorch
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

        #CHECKING:

        #attempting to put conv layer over rnn
        # self.conv = nn.Sequential(
        #     nn.Conv1d(4, 64, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=4),  # 2048 → 512
        #     nn.Dropout(0.1)
        # )
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Set initial states

        #just trying to place conv layer on top
        # x = x.permute(0, 2, 1)  # [B, 2048, 4] -> [B, 4, 2048]
        # x = self.conv(x)  # [B, 4, 2048] -> [B, 64, 512]
        # x = x.permute(0, 2, 1)  # [B, 64, 512] -> [B, 512, 64]

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step

        #this might make or break model b/c trained like this.
        #not sure if attention pooling would be better - attempting...
        #becuase of high values of CAGE.

        #TESTING - ATTENTION POOLING
        attn = torch.softmax(self.attention(out), dim=1)
        out = (out * attn).sum(dim=1)
        #TEST END

        out = self.dropout(out)  #.mean(dim=1)
        out = self.fc(out)
        return out

def rnn_model():
    torch.cuda.set_device(0)
    #get gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    #puts model to gpu
    model = BidLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5) #prev trial : 0.001

    #if learning stagnates lower rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40
    )
    loss = WeightedHuberLoss(label_mean, label_std)
    #trying MSE loss first and then huber to see if more accurate than CNN
    #Huber loss may be better because there are definitely outliers in the data
    training_loader = torch.utils.data.DataLoader(training_set,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        num_workers=0)
    val_set = dataset['validation']
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0
                                             )
    # print(iter(training_loader))
    print(f"Starting training for {epochs} epochs...\n")
    best_loss = float('inf')
    patience_counter = 0
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

            # labels = (labels - label_mean) / label_std try log
            labels = torch.log1p(labels)
            optimizer.zero_grad()

            outputs = model(sequences).view(-1, 16, 50)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                if batch_idx % 100 == 0:
                    print(f"Skipped batch {batch_idx} - NaN outputs")
                optimizer.zero_grad()
                continue
            # add outputs with labels to our loss
            batch_loss = loss(outputs, labels)
            if torch.isnan(batch_loss) or torch.isinf(batch_loss) or batch_loss.item() > 1.5:
                optimizer.zero_grad()
                continue

            epoch_loss += batch_loss.item()
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.8)

            optimizer.step()

            if batch_idx % 100 == 0:
                with torch.no_grad():
                    pred_std = outputs.std().item()
                    pred_mean = outputs.mean().item()
                global_step = epoch * len(training_loader) + batch_idx
                writer.add_scalar("Loss/batch", batch_loss.item(), global_step)
                writer.add_scalar("Predictions/std", pred_std, global_step)
                writer.add_scalar("Predictions/mean", pred_mean, global_step)
                writer.flush()
                print(f"Epoch {epoch + 1}/{epochs} | Batch "
                      f"{batch_idx}/{len(training_loader)} "
                      f"| Batch Loss: {batch_loss.item():.4f} "
                      f"| Pred std: {pred_std:.4f}")

                if pred_std < 0.01:
                    print("Model collapsing! Predictions too constant")


        avg_loss = epoch_loss / len(training_loader)
        # last_train_outputs = outputs.detach()
        # last_train_labels = labels

        model.eval()
        val_loss = 0
        #start checking validation loss for ---
        with torch.no_grad():
            for batch in val_loader:
                sequences = torch.stack([encoding(s).permute(1, 0) for s in batch["sequence"]]).to(device)
                labels_np = np.array(batch["labels"])
                if labels_np.shape != (labels_np.shape[0], 16, 50):
                    labels_np = labels_np.transpose(2, 0, 1)
                labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
                # labels = (labels - label_mean) / label_std
                labels = torch.log1p(labels) #try log
                outputs = model(sequences).view(-1, 16, 50)

                val_loss += loss(outputs, labels).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']

        #SCHEDULER CHECK ___________________________
        patience = 60
        if val_loss < best_loss - 0.0005:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "RNN_testline/rnn_best4checkptbest_current.pth")
            print(f"  New best!")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch}. Best: {best_loss:.4f}")
            break
        #SCHEDULER CHECK _________________________________


        print(f"  Scheduler check | LR: {current_lr} | patience_counter: {patience_counter}/{patience}")
        writer.add_scalar("Loss/val", val_loss, epoch)
        #stopped copy pasted code ---
        #scheduler.step(avg_loss)

        if not torch.isnan(outputs).any():

            writer.add_histogram("Predictions/distribution", outputs.detach(), epoch)
            writer.add_histogram("Labels/distribution", labels, epoch)
        else:
            print(f"Skipping histogram at epoch {epoch} — NaN outputs detected")
        writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        for name, param in [("LSTM/weights(extra)", model.lstm.weight_ih_l0),
                            ("FC/weights(extra)", model.fc.weight)]:
            if not torch.isnan(param).any() and not torch.isinf(param).any():
                writer.add_histogram(name, param, epoch)
        if epoch == 99:
            #used ai for writer just for easier time
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            opt_name = type(optimizer).__name__  # e.g. "Adam"
            checkpoint_name = f"rnn_baseline_epoch{epoch}_{timestamp}_{opt_name}"
            writer_mid = SummaryWriter(log_dir=f"runs/{checkpoint_name}")
            torch.save(model.state_dict(), f"RNN_testline/{checkpoint_name}.pth")
            writer_mid.close()

        print(f"Epoch {epoch + 1}/{epochs}.. " + f" | {avg_loss}")
        print(f"Epoch {epoch + 1}/{epochs}.. " + f" | {val_loss}")
    writer.close()
    torch.save(model.state_dict(), f"RNN_testline/rnn_baseline_fixed150.pth")
    torch.save(optimizer.state_dict(), f"RNN_testline/optimizer{epochs}.pth")
    print("Training complete! saved to rnn_baseline.pth")
    return model

#ai assisted func - unused curr
def continue_until_convergence():
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model from epoch 50
    model = BidLSTM().to(device)
    model.load_state_dict(torch.load("RNN_testline/rnn_best3checkpt250ep.pth", map_location=device))
    model.eval()
    eval_loss = 0

    with torch.no_grad():
        test_seq = encoding(training_set[0]['sequence']).permute(1, 0).unsqueeze(0).to(device)
        test_out = model(test_seq).view(-1, 16, 50)
        print(f"check pred mean: {test_out.mean():.4f}, std: {test_out.std():.4f}")



    # high lr from 0.003 at stage 50
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) #learning rate on 50: .0003 - 100-190 epochs

    #attempting with .0001 to figure out convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40
    )

    loss_fn = WeightedHuberLoss(label_mean, label_std)

    training_loader = torch.utils.data.DataLoader(training_set,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

    # Early stopping
    best_loss = 1.128 #prev one
    patience = 2
    patience_counter = 0
    start_epoch = 200
    print(f" Loaded model from epoch f{start_epoch}")

    max_epochs = 230

    run_name_continue = f"rnn_continue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer_continue = SummaryWriter(log_dir=f"runs/{run_name_continue}")
    # eval_loss = 0
    # with torch.no_grad():
    #     for batch in training_loader:
    #         sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)
    #         labels_np = np.array(batch["labels"])
    #         if labels_np.shape != (batch_size, 16, 50):
    #             labels_np = labels_np.transpose(2, 0, 1)
    #         labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
    #         labels = (labels - label_mean) / label_std
    #         outputs = model(sequences).view(-1, 16, 50)
    #         eval_loss += loss_fn(outputs, labels).item()
    # reconstructed_loss = eval_loss / len(training_loader)
    # writer_continue.add_scalar("Loss/epoch", reconstructed_loss, 100)  # ← add this
    # print(f"Reconstructed epoch 100 loss: {reconstructed_loss:.4f}")  # ← and this
    # print(f"\n Continuing from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()

        epoch_loss = 0

        for batch_idx, batch in enumerate(training_loader):
            sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)
            labels_np = np.array(batch["labels"])
            if labels_np.shape != (batch_size, 16, 50):
                labels_np = labels_np.transpose(2, 0, 1)

            labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
            # labels = (labels - label_mean) / label_std
            labels = torch.log1p(labels) #try log
            optimizer.zero_grad()
            outputs = model(sequences).view(-1, 16, 50)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                if batch_idx % 100 == 0:
                    print(f"Skipped batch {batch_idx} - NaN outputs")
                optimizer.zero_grad()
                continue
            batch_loss = loss_fn(outputs, labels)
            #epoch_loss += batch_loss.item()
            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                print(f" NaN/Inf at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if batch_idx % 100 == 0:
                global_step = (epoch - start_epoch) * len(training_loader) + batch_idx
                with torch.no_grad():
                    pred_std = outputs.std().item()
                    pred_mean = outputs.mean().item()
                    pred_max = outputs.max().item()
                    pred_min = outputs.min().item()
                    if not torch.isnan(outputs).any():


                        #extra, really unneeded I think
                        pred_img_tensor = outputs[0].detach().cpu().unsqueeze(0)  # [1, 16, 50]
                        pred_img_tensor = (pred_img_tensor - pred_img_tensor.min()) / \
                                          (pred_img_tensor.max() - pred_img_tensor.min() + 1e-8)
                        label_img_numpy = labels[0].cpu().numpy()  # [16, 50]
                        label_img_numpy = (label_img_numpy - label_img_numpy.min()) / \
                                          (label_img_numpy.max() - label_img_numpy.min() + 1e-8)
                        label_img_numpy = label_img_numpy[np.newaxis, :, :]
                        writer_continue.add_image("Images/prediction_tensor_grid", pred_img_tensor, global_step)
                        writer_continue.add_image("Images/label_numpy_tr_gridCHW",
                                                  label_img_numpy, global_step, dataformats='CHW')
                    # ADD TENSORBOARD LOGGING:

                writer_continue.add_scalar("Loss/batch", batch_loss.item(), global_step)
                writer_continue.add_scalar("Predictions/std", pred_std, global_step)
                writer_continue.add_scalar("Predictions/mean", pred_mean, global_step)
                writer_continue.add_scalar("Predictions/max", pred_max, global_step)
                writer_continue.add_scalar("Predictions/min", pred_min, global_step)
                writer_continue.add_histogram("Distributions", outputs, global_step)
                #img test

                # label same way - not going to lie - used ai to help write the image labels cause
                #was confused for a moment. not great in practice but am trying to rerun fast


                print(f"Epoch {epoch}/{max_epochs} | Batch {batch_idx} | "
                      f"Loss: {batch_loss.item():.4f} | Std: {pred_std:.4f}")
                if epoch == 10:
                    print(model.eval())
                if epoch == 25:
                    print(model.state_dict())
                    print(model.eval())
        avg_loss = epoch_loss / len(training_loader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Scheduler check | LR: {current_lr} | patience_counter: {patience_counter}/{patience}")
        writer_continue.add_scalar("Loss/epoch", avg_loss, epoch)
        writer_continue.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        print(f"Epoch {epoch}/{max_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 0.001:
            prev_best = best_loss
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), "RNN_testline/rnn_best4checkptbest_current.pth")
            print(f"  New best!")
            print(f"  New best! {prev_best:.4f} → {best_loss:.4f}")
            print(f"Epoch {epoch}/{max_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"\n Early stopping at epoch {epoch}. Best: {best_loss:.4f}")
            break

    writer_continue.close()
    print(" Done! Saved to model: rnn_best.pth")
    return model
# # def calculateTSS(model, val_split):
# #     model.eval()
# #     total_recall = []
# #     total_precision = []
# #     #redundancy
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     with torch.no_grad():
# #         for batch_idx, batch in enumerate(val_split):
# #             sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)
# #             labels_np = np.array(batch["labels"])
# #             if labels_np.shape != (batch_size, 16, 50):
# #                 labels_np = labels_np.transpose(2, 0, 1)
# #             outputs = model(sequences).view(-1, 16, 50)
# #             preds_denorm = (outputs.cpu().numpy() * label_std + label_mean)
# #             labels_denorm = labels_np * label_std + label_mean
# #
# #             for pred, actual in zip(preds_denorm, labels_denorm):
# #                 pred_flat = pred.flatten()
# #                 actual_flat = actual.flatten()
# #
# #                 pred_peaks, _ = find_peaks(pred_flat, height=1.0)
# #                 actual_peaks, _ = find_peaks(actual_flat, height=1.0)
# #
#     return model
def evaluate_rnn(model, test_set, device):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0
    loss_fn = WeightedHuberLoss(label_mean, label_std)
    all_preds, all_labels_out = [], []

    with torch.no_grad():
        for batch in test_loader:
            sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)
            labels_np = np.array(batch["labels"])
            actual_bs = labels_np.shape[0]  # use actual batch size, not hardcoded
            if labels_np.shape != (actual_bs, 16, 50):
                labels_np = labels_np.transpose(2, 0, 1)
            labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
            # labels = (labels - label_mean) / label_std
            labels = torch.log1p(labels) # try log
            outputs = model(sequences).view(-1, 16, 50)
            total_loss += loss_fn(outputs, labels).item()
            #testing r just in case even though metric doesnt matter much since non-lin
            all_preds.append(outputs.cpu().numpy().flatten())
            all_labels_out.append(labels.cpu().numpy().flatten())

    avg_test_loss = total_loss / len(test_loader)
    r, _ = pearsonr(np.concatenate(all_preds), np.concatenate(all_labels_out))
    rho, _ = spearmanr(np.concatenate(all_preds), np.concatenate(all_labels_out))
    print(f"Pearson R:  {r:.4f}")
    print(f"Spearman R: {rho:.4f}  (more reliable for CAGE)")
    print(f"Test loss (normalized): {avg_test_loss:.4f}")
    return avg_test_loss


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # import glob
    #
    # checkpoints = glob.glob("RNN_testline/*.pth")
    #
    # results = {}
    # for ckpt_path in checkpoints:
    #     ckpt_name = os.path.basename(ckpt_path)
    #     try:
    #         model = BidLSTM().to(device)
    #         model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #         loss = evaluate_rnn(model, test_set, device)
    #         results[ckpt_name] = loss
    #         print(f"{ckpt_name}: {loss:.4f}")
    #     except Exception as e:
    #         print(f"{ckpt_name}: FAILED ({e})")
    #
    # print("\n--- RANKING ---")
    # for name, loss in sorted(results.items(), key=lambda x: x[1]):
    #     print(f"{loss:.4f}  {name}")
    os.makedirs("RNN_testline", exist_ok=True)
    #model = continue_until_convergence()
    model = rnn_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate_rnn(model, test_set, device)
