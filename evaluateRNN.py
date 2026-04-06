import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from datasets import load_dataset
from scipy.signal import find_peaks


#RUN IN TERMINAL
print("=" * 60)
print("RNN BASELINE EVALUATION")
print("=" * 60)

label_mean = 0.1963
label_std = 4.7285

"""
testing
import glob

checkpoints = glob.glob("RNN_testline/*.pth")
results = {}
for ckpt_path in checkpoints:
    ckpt_name = os.path.basename(ckpt_path)
    try:
        model = BidLSTM().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        loss = evaluate_rnn(model, test_set, device)
        results[ckpt_name] = loss
        print(f"{ckpt_name}: {loss:.4f}")
    except Exception as e:
        print(f"{ckpt_name}: FAILED ({e})")

print("\n rank:")
for ckpt_path in checkpoints:
    ckpt_name = os.path.basename(ckpt_path)
for name, loss in sorted(results.items(), key=lambda x: x[1]):
    print(f"{loss:.4f}  {name}")
"""

# ========== DEFINE RNN CLASS ==========
class BidLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=2, num_classes=800):
        super(BidLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.15)
        self.dropout = nn.Dropout(0.27)  # signs of overfitting
        # https://codesignal.com/learn/courses -- good resource
        # /improving-neural-networks-with-pytorch/lessons/adding-dropout-to-neural-networks-in-pytorch
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Multiply by 2 because of bidirectional

        # CHECKING:

        # attempting to put conv layer over rnn
        # self.conv = nn.Sequential(
        #     nn.Conv1d(4, 64, kernel_size=7, padding=3),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=4, stride=4),  # 2048 → 512
        #     nn.Dropout(0.1)
        # )
        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # Set initial states

        # just trying to place conv layer on top
        # x = x.permute(0, 2, 1)  # [B, 2048, 4] -> [B, 4, 2048]
        # x = self.conv(x)  # [B, 4, 2048] -> [B, 64, 512]
        # x = x.permute(0, 2, 1)  # [B, 64, 512] -> [B, 512, 64]

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # 2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step

        # this might make or break model b/c trained like this.
        # not sure if attention pooling would be better - attempting...
        # becuase of high values of CAGE.

        # TESTING - ATTENTION POOLING
        attn = torch.softmax(self.attention(out), dim=1)
        out = (out * attn).sum(dim=1)
        # TEST END

        out = self.dropout(out)  # .mean(dim=1)
        out = self.fc(out)
        return out
# ========== ENCODING FUNCTION ==========
def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq], dtype=torch.long)
    one_hot = torch.zeros(4, len(seq))
    one_hot[indices, torch.arange(len(seq))] = 1
    return one_hot


# ========== LOAD DATASET ==========
print("\nLoading test dataset...")
dataset = load_dataset(
    "InstaDeepAI/genomics-long-range-benchmark",
    task_name="cage_prediction",
    sequence_length=2048,
    trust_remote_code=True,
    cache_dir="C:/genomics"
)
test_set = dataset["test"]
print(f"Test set: {len(test_set)} examples")

# ========== LOAD MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = BidLSTM()

model.load_state_dict(torch.load("RNN_testline/rnn_best4checkptbest_current.pth", map_location=device))
# model.load_state_dict(torch.load("past_runs/62EpochGoodResults/", map_location=device))

model = model.to(device)
model.eval()

print(" Model loaded successfully\n")

# ========== EVALUATE ==========
print("Model loaded successfully\n")

# ========== DIAGNOSTIC ==========
print("CHECKING IF MODEL OUTPUTS VARY:")

test_sample_1 = test_set[0]
test_sample_2 = test_set[100]

seq1 = encoding(test_sample_1['sequence']).permute(1, 0).unsqueeze(0).to(device)
seq2 = encoding(test_sample_2['sequence']).permute(1, 0).unsqueeze(0).to(device)
with torch.no_grad():
    out1 = model(seq1)
    out2 = model(seq2)

print(f"Sample 1 output - Mean: {out1.mean():.6f}, Std: {out1.std():.6f}")
print(f"Sample 2 output - Mean: {out2.mean():.6f}, Std: {out2.std():.6f}")
print(f"Difference between samples: {(out1 - out2).abs().mean():.6f}")

if (out1 - out2).abs().mean() < 0.001:
    print("WARNING: Model outputs are nearly identical for different inputs!")
    print("   Model has COLLAPSED to predicting constant values!")
else:
    print("Model outputs vary (good)")

print("=" * 60)
print()

# ========== EVALUATE ==========
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

all_preds = []
all_labels = []
total_loss = 0
loss_fn = nn.HuberLoss(delta=1)

print("Evaluating...")

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        actual_batch_size = len(batch["sequence"])
        sequences = torch.stack([encoding(seq).permute(1, 0) for seq in batch["sequence"]]).to(device)

        labels_np = np.array(batch["labels"])
        if labels_np.shape != (actual_batch_size, 16, 50):
            labels_np = labels_np.transpose(2, 0, 1)

        labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
        labels = torch.log1p(labels)

        outputs = model(sequences).view(-1, 16, 50)
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(test_loader)}")

print("Done!\n")

# ========== METRICS ==========
preds = np.concatenate(all_preds).flatten()
labels = np.concatenate(all_labels).flatten()

test_mse = total_loss / len(test_loader)
test_mae = mean_absolute_error(labels, preds)
r2 = r2_score(labels, preds)
pearson_r = pearsonr(preds, labels)[0]
spearman_r = spearmanr(preds, labels)[0]

print("=" * 60)
print("RNN BASELINE TEST RESULTS")
print("=" * 60)
print(f"  Test MSE:      {test_mse:.4f}")
print(f"  Test MAE:      {test_mae:.4f}")
print(f"  R² Score:      {r2:.4f}")
print(f"  Pearson R:     {pearson_r:.4f}")
print(f"  Spearman R:    {spearman_r:.4f}")
print("=" * 60)

if pearson_r > 0.4:
    print("\nEXCELLENT!")
elif pearson_r > 0.3:
    print("\nGOOD baseline.")
elif pearson_r > 0.1:
    print("\nMEDIOCRE but usable.")
else:
    print("\nmodel not linear.")

idx = np.random.choice(len(preds), 5000)
plt.scatter(labels[idx], preds[idx], alpha=0.1, s=1)
plt.xlabel('Actual CAGE (log1p)')
plt.ylabel('Predicted CAGE (log1p)')
plt.title('RNN: Predicted vs Actual CAGE')
#plt.plot([0, 5], [0, 5], 'r--', label='Perfect prediction')
m, b = np.polyfit(labels[idx], preds[idx], 1)
plt.plot(labels[idx], m * labels[idx] + b, 'r-', label=f'Line of best fit')
plt.legend()
plt.savefig('rnn_scatter.png', dpi=150)