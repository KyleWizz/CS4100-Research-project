import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error
from datasets import load_dataset
from neuralop.models import FNO
#RUN ALL ON TERMINAL in venv
print("=" * 60)
print("FNO BASELINE EVALUATION")
print("=" * 60)

def encoding(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    indices = torch.tensor([mapping.get(base, 0) for base in seq], dtype=torch.long)
    one_hot = torch.zeros(4, len(seq))
    one_hot[indices, torch.arange(len(seq))] = 1
    return one_hot

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
            n_modes=(32,),  # was 64
            in_channels=4,
            out_channels=16,  # was 32
            hidden_channels=32,  # was 64
            n_layers=3,  # was 4
        )
        self.pool = nn.AdaptiveAvgPool1d(50)  # [batch, 32, 25]
        # using staack
        self.stack = nn.Sequential(
            nn.Linear(16 * 50, 256),  # 16 not 32
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, 800)
        )
        # self.stack = nn.Sequential(
        #     nn.Linear(32 * 25, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.35),
        #     nn.Linear(256, 800)
        # )

    def forward(self, x):
        # x: [batch, 4, 2048] — no permute needed unlike RNN
        out = self.fno(x)  # [batch, 32, 2048]
        out = self.pool(out)  # [batch, 32, 25]
        out = out.flatten(1)  # [batch, 800]
        out = self.stack(out)  # [batch, 800]
        return out


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = FNO_class().to(device)
#model.load_state_dict(torch.load("FNO_testline/fno_current_bestuse.pth", map_location=device))
state_dict = torch.load("FNO_testline/fno_current_bestuse.pth", map_location=device)
state_dict.pop("_metadata", None)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully\n")

# diagnostic
test_sample_1 = test_set[0]
test_sample_2 = test_set[100]
seq1 = encoding(test_sample_1['sequence']).unsqueeze(0).to(device)
seq2 = encoding(test_sample_2['sequence']).unsqueeze(0).to(device)
with torch.no_grad():
    out1 = model(seq1)
    out2 = model(seq2)
print(f"Sample 1 - Mean: {out1.mean():.6f}, Std: {out1.std():.6f}")
print(f"Sample 2 - Mean: {out2.mean():.6f}, Std: {out2.std():.6f}")
print(f"Difference: {(out1 - out2).abs().mean():.6f}")
if (out1 - out2).abs().mean() < 0.001:
    print("WARNING: Model collapsed!")
else:
    print("Model outputs vary (good)")
print("=" * 60)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
all_preds, all_labels = [], []
total_loss = 0
loss_fn = nn.HuberLoss(delta=1)

print("Evaluating...")
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        actual_bs = len(batch["sequence"])
        sequences = torch.stack([encoding(seq) for seq in batch["sequence"]]).to(device)
        labels_np = np.array(batch["labels"])
        if labels_np.shape != (actual_bs, 16, 50):
            labels_np = labels_np.transpose(2, 0, 1)
        labels = torch.from_numpy(labels_np.astype(np.float32)).to(device)
        labels = torch.log1p(labels)
        outputs = model(sequences).view(-1, 16, 50)
        total_loss += loss_fn(outputs, labels).item()
        all_preds.append(outputs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(test_loader)}")

preds = np.concatenate(all_preds).flatten()
labels = np.concatenate(all_labels).flatten()
pearson_r = pearsonr(preds, labels)[0]
spearman_r = spearmanr(preds, labels)[0]
r2 = r2_score(labels, preds)
mae = mean_absolute_error(labels, preds)

print("=" * 60)
print("FNO BASELINE TEST RESULTS")
print("=" * 60)
print(f"  Test Loss:     {total_loss / len(test_loader):.4f}")
print(f"  Test MAE:      {mae:.4f}")
print(f"  R² Score:      {r2:.4f}")
print(f"  Pearson R:     {pearson_r:.4f}")
print(f"  Spearman R:    {spearman_r:.4f}  <- most reliable for CAGE")
print("=" * 60)
import matplotlib.pyplot as plt
import numpy as np

# after running eval, you have all_preds and all_labels
preds = np.concatenate(all_preds).flatten()
labels = np.concatenate(all_labels).flatten()

# sample 5000 points so it's not too dense
idx = np.random.choice(len(preds), 5000)
plt.scatter(labels[idx], preds[idx], alpha=0.1, s=1)
plt.xlabel('Actual CAGE (log1p)')
plt.ylabel('Predicted CAGE (log1p)')
plt.title('FNO: Predicted vs Actual CAGE')
#plt.plot([0, 5], [0, 5], 'r--', label='Perfect prediction')
m, b = np.polyfit(labels[idx], preds[idx], 1)
plt.plot(labels[idx], m * labels[idx] + b, 'r-', label=f'Line of best fit')
plt.legend()
plt.savefig('fno_scatter.png', dpi=150)