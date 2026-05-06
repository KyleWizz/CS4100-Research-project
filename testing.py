from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr #just checking

import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from neuralop.models import FNO
from torch.utils.data import TensorDataset

# some libs imported in case
#https://neuraloperator.github.io/dev/theory_guide/fno.html docs
# RUN IN TERMINAL
""" 
(.venv) PS C:  Users \ User\PycharmProjects\PythonProject\CS4100-Research-project> C:
 Users \ User \PycharmProjects\PythonProject\.venv\Scripts\python.exe .\FNO_baseline.py                                                                                 
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
sequence_length = 10240
# trained on 2048 bp len sequence (ACTGN)

# task_name = "variant_effect_causal_eqtl"
# task_name = "bulk_rna_expression"
# vector of 218 different tissue types - sequence outputs the same vector matrix [218]

task_name = "cage_prediction"
run_name = f"{datetime.now().strftime('FNO_run-%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir=f"runs/{run_name}")
# One of:
# ["variant_effect_causal_eqtl","variant_effect_pathogenic_clinvar",
# "variant_effect_pathogenic_omim","cage_prediction", "bulk_rna_expression",
# "chromatin_features_histone_marks","chromatin_features_dna_accessibility",
# "regulatory_element_promoter","regulatory_element_enhancer"]

# goal is to do bulk rna expression eventually, possibly after my project.
# build a CNN baseline -> RNN baseline -> FNO baseline and compare.
# features to have after these three steps : UI and graphs, research paper attributed to this repo.
# currently working on cage prediction as it may be quicker, can continue with bulk later on.

# dataset = load_dataset(
# "InstaDeepAI/genomics-long-range-benchmark", "bulk_rna_expression",
# dataset load was broken so messed with it - data set 2.19, pyfaidx downloaded, os path changed
os.makedirs("C:/genomics/downloads", exist_ok=True)
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name=task_name,
    sequence_length=sequence_length,
    trust_remote_code=True,
    cache_dir="C:/genomics"
    # b_rna_ex outputs 218 - might have to change to work - still WIP
    # seq. wise regression
    # bulk_rna_expression = bulk_rna_expression,
    # subset = True, if applicable - for now instead of training on millions, subset for computational time-sake
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
# since it takes a little!
epochs = 200
batch_size = 96

print(training_set)

print(test_set)


# one-hot encode nucleotide sequences
def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq])
    one_hot = torch.zeros(4, len(seq))  # [4, 2048]
    #one_hot[indices, torch.arange(len(seq))] = 1
    one_hot.scatter_(0, indices.unsqueeze(0), 1)
    return one_hot


def encode_dataset(hf_dataset):
    print("Encoding...")
    seqs = torch.stack([encoding(s) for s in hf_dataset["sequence"]])
    labels_np = np.array(hf_dataset["labels"])
    # always transpose from [N, 50, 80] → [N, 80, 50]
    if labels_np.ndim == 3 and labels_np.shape[1] == 50:
        labels_np = labels_np.transpose(0, 2, 1)
    print(f"dimensions: {labels_np.ndim}")
    labels = torch.from_numpy(labels_np.astype(np.float32))
    labels = torch.log1p(labels)
    return TensorDataset(seqs, labels)

train_dataset = encode_dataset(training_set)
val_dataset   = encode_dataset(dataset['validation'])
test_dataset  = encode_dataset(test_set)

# data seems to have some outliers, so weight as well
# CAGE signaling has lots of potential noise, read more:
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

    def forward(self, pred, target):
        weights = torch.ones_like(target)
        #testing with normalized

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

# debugging
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
FNOmodel begin
"""


class FNO_class(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fno = FNO(
        #     n_modes=(32,),
        #     in_channels=4,
        #     out_channels=32,
        #     hidden_channels=64,
        #     n_layers=4,
        # )
        self.fno = FNO(
            n_modes=(64,),  # was 64
            in_channels=4,
            out_channels=32,  # was 32
            hidden_channels=64,  # was 64
            n_layers=3,  # was 4
        )
        self.pool = nn.AdaptiveAvgPool1d(50)  # [batch, 32, 25]
        #using staack
        self.stack = nn.Sequential(
            nn.Linear(32 * 50, 256),  # 16 not 32
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 4000)
        )
        # self.stack = nn.Sequential(
        #     nn.Linear(32 * 25, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.35),
        #     nn.Linear(256, 800)
        # )

    def forward(self, x):
        # x: [batch, 4, 2048] — no permute needed unlike RNN
        with torch.autocast(device_type="cuda", enabled=False):
            out = self.fno(x.float())  # [batch, 32, 2048]
        out = self.pool(out)    # [batch, 32, 25]
        out = out.flatten(1)    # [batch, 800]
        out = self.stack(out)      # [batch, 800]
        return out

def FNO_model():
    torch.cuda.set_device(0)
    # get gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # puts model to gpu
    model = FNO_class().to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)  # prev trial : 0.001

    # if learning stagnates lower rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=35
    )
    scaler = torch.amp.GradScaler('cuda')
    # loss = WeightedHuberLoss(label_mean, label_std)
    """
    TESTING WITH POISSON LOSS FOR MORE ACCURATE RESULTS - WEIGHTED HUBER LOSS WAS USED
    TO GET RESULTS
    """
    loss = torch.nn.PoissonNLLLoss(reduction='mean', eps=1e-8, full=True)
    # # trying MSE loss first and then huber to see if more accurate than CNN
    # # Huber loss may be better because there are definitely outliers in the data
    # training_loader = torch.utils.data.DataLoader(training_set,
    #                                               batch_size=batch_size,
    #                                               shuffle=True,
    #                                               num_workers=0,
    #                                               pin_memory=True,
    #                                               )
    # val_set = dataset['validation']
    # val_loader = torch.utils.data.DataLoader(val_set,
    #                                          batch_size=batch_size,
    #                                          shuffle=False,
    #                                          num_workers=0
    #                                          )
    # # print(iter(training_loader))

    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Starting training for {epochs} epochs...\n")
    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        # for batch_idx, batch in enumerate(training_loader):
        #     # prep
        #     sequences = torch.stack([encoding(seq) for seq in batch["sequence"]]).to(device)
        #     labels_np = np.array(batch["labels"])
        #     if labels_np.shape != (batch_size, 80, 50):
        #         labels_np = labels_np.transpose(2, 0, 1)
        #     labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
        #     labels = torch.log1p(labels)
        for batch_idx, (sequences, labels) in enumerate(training_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(sequences).view(-1, 80, 50)
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    if batch_idx % 100 == 0:
                        print(f"Skipped batch {batch_idx} - NaN outputs")
                    continue
                batch_loss = loss(outputs, labels)






            if torch.isnan(batch_loss) or torch.isinf(batch_loss) or batch_loss.item() > 1.0:
                continue
            epoch_loss += batch_loss.item()
            # batch_loss.backward()
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
            scaler.step(optimizer)
            scaler.update()
            #optimizer.step()
            if batch_idx % 100 == 0:
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

        avg_loss = epoch_loss / len(training_loader)
        """
        OPTIMIZER AND SCHEDULER BLOCK
        """



        model.eval()

        val_loss = 0
        # start checking validation loss for ---

        with torch.no_grad():
            for (sequences, labels) in val_loader:
                # sequences = torch.stack([encoding(s) for s in batch["sequence"]]).to(device)
                # labels_np = np.array(batch["labels"])
                # if labels_np.shape != (labels_np.shape[0], 80, 50):
                #     labels_np = labels_np.transpose(2, 0, 1)
                # labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
                # labels = torch.log1p(labels)
                # outputs = model(sequences).view(-1, 80, 50)
                # val_loss += loss(outputs, labels).item()
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences).view(-1, 80, 50)
                val_loss += loss(outputs, labels).item()
            val_loss /= len(val_loader)
            scheduler.step(val_loss)

            #stop val loss ---- - on val loss may be better than
            #the traning step loss (avg loss)

            current_lr = optimizer.param_groups[0]['lr']

            # SCHEDULER CHECK ___________________________
            max_epochs = 200
            patience = 60
            if val_loss < best_loss - 0.0005:
                prev_best = best_loss
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "FNO_testline/fno_current_bestuse.pth")
                print(f"  New best!")
                print(f"  New best! {prev_best:.4f} → {best_loss:.4f}")
                print(f"Epoch {epoch}/{max_epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\n Early stopping at epoch {epoch}. Best: {best_loss:.4f}")
                break
            # SCHEDULER CHECK _________________________________

            print(f"  Scheduler check | LR: {current_lr} | patience_counter: {patience_counter}/{patience}")
            writer.add_scalar("Loss/val", val_loss, epoch)
            # stopped copy pasted code ---
            # scheduler.step(avg_loss)

            if not torch.isnan(outputs).any():

                writer.add_histogram("Predictions/distribution", outputs.detach(), epoch)
                writer.add_histogram("Labels/distribution", labels, epoch)
            else:
                print(f"Skipping histogram at epoch {epoch} — NaN outputs detected")
            writer.add_scalar("Loss/epoch_avg", avg_loss, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)

            if epoch == 99:
                # used ai for writer just for easier time
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                opt_name = type(optimizer).__name__  # e.g. "Adam"
                checkpoint_name = f"FNO_baseline_epoch{epoch}_{timestamp}_{opt_name}"
                writer_mid = SummaryWriter(log_dir=f"runs/{checkpoint_name}")
                torch.save(model.state_dict(), f"FNO_testline/{checkpoint_name}.pth")
                writer_mid.close()

            print(f"Epoch {epoch + 1}/{epochs}.. " + f" | avg loss: | {avg_loss} "
                                                     f"| val_loss: | {val_loss}")

    writer.close()
    torch.save(model.state_dict(), f"FNO_testline/FNO_baseline_fixed{epochs}.pth")
    torch.save(optimizer.state_dict(), f"FNO_testline/optimizer{epochs}.pth")
    print("Training complete! saved to rnn_baseline.pth")
    return model

def eval_fno(model, test_set, device):
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0
    loss_fn = WeightedHuberLoss(label_mean, label_std)
    all_preds, all_labels_out = [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            # sequences = torch.stack([encoding(seq) for seq in batch["sequence"]]).to(device)
            sequences = sequences.to(device)
            labels_np = labels.to(device)

            # labels_np = np.array(batch["labels"])
            actual_bs = labels_np.shape[0]
            if labels_np.shape != (actual_bs, 80, 50):
                labels_np = labels_np.transpose(2, 0, 1)
            labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
            labels = torch.log1p(labels)
            outputs = model(sequences).view(-1, 80, 50)
            total_loss += loss_fn(outputs, labels).item()
            #testing r just in case even though metric doesnt matter much since non-lin
            all_preds.append(outputs.cpu().numpy().flatten())
            all_labels_out.append(labels.cpu().numpy().flatten())

    avg_test_loss = total_loss / len(test_loader)
    r, _ = pearsonr(np.concatenate(all_preds), np.concatenate(all_labels_out))
    #for paper comparison
    rho, _ = spearmanr(np.concatenate(all_preds), np.concatenate(all_labels_out))
    print(f"Pearson R:  {r:.4f}")
    print(f"Spearman R: {rho:.4f}  (more reliable for CAGE)")
    print(f"Test loss (normalized): {avg_test_loss:.4f}")
    return avg_test_loss

if __name__ == '__main__':
    os.makedirs("FNO_testline", exist_ok=True)
    model = FNO_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_fno(model, test_set, device)

