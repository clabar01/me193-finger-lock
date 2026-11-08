"""
ME193 Finger Lock, Part 1: fitting the V data from Anna's Code with gradient descent.

Part A: a single neuron  y_hat = m*u + b  trained with hand-written gradient descent
        (the exact same math as v_fit.xlsx). It can only make a straight line, so it
        can't fit a V and the loss levels off.
Part B: a hidden layer of 2 ReLU neurons + a linear output neuron, built in PyTorch.
        ReLU bends the line, so the two neurons make the two arms of the V.

Run from the repo root:   python part1_v_fit/v_fit.py
Outputs (saved next to this file):
    single_neuron.png   data + fit, and loss vs epoch, for Part A
    relu_network.png    data + fit, and loss vs epoch, for Part B
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
DATA = HERE / "annacode-v.tsv"


# ---------------------------------------------------------------- load data
def load_data():
    """Anna's Code export: columns are time, x, y (tab separated, 1 header line)."""
    d = np.loadtxt(DATA, delimiter="\t", skiprows=1)
    x, y = d[:, 1], d[:, 2]
    order = np.argsort(x)          # sort by x (same as the spreadsheet)
    return x[order], y[order]


x, y = load_data()
N = len(x)

# Normalize x, the same trick as the spreadsheet and Anna's Code.
# Raw x runs 20 to 100, which makes the m gradient way bigger than the b gradient.
x_mean, x_std = x.mean(), x.std()
u = (x - x_mean) / x_std


# ================================================================ Part A
def single_neuron_gd(u, y, m, b, alpha, epochs):
    """Plain gradient descent on y_hat = m*u + b with mean squared error.

    dL/dm = (2/N) * sum((y_hat - y) * u)
    dL/db = (2/N) * sum(y_hat - y)
    """
    n = len(u)
    history = []
    for _ in range(epochs):
        y_hat = m * u + b                     # 1. run every point through the neuron
        err = y_hat - y                       # 2. how far off each prediction is
        mse = np.mean(err ** 2)               # 3. mean squared error (the loss)
        dm = 2 / n * np.sum(err * u)          # 4. gradients from the chain rule
        db = 2 / n * np.sum(err)
        history.append((m, b, mse, dm, db))
        m = m - alpha * dm                    # 5. step downhill
        b = b - alpha * db
    return m, b, history


# Check against the spreadsheet: same 10 points (every 12th), same start, same alpha.
idx10 = np.arange(5, N, 12)[:10]
u10 = (x[idx10] - x[idx10].mean()) / x[idx10].std()
_, _, check = single_neuron_gd(u10, y[idx10], m=10.0, b=0.0, alpha=0.1, epochs=5)
print("Excel check (10 points, should match v_fit.xlsx rows 20-24):")
print(f"{'iter':>4} {'m':>9} {'b':>9} {'MSE':>10} {'dL/dm':>9} {'dL/db':>9}")
for i, (m_, b_, mse_, dm_, db_) in enumerate(check, start=1):
    print(f"{i:>4} {m_:9.3f} {b_:9.3f} {mse_:10.2f} {dm_:9.3f} {db_:9.3f}")

# Now train on all 120 points for longer.
EPOCHS_A = 100
m, b, hist_a = single_neuron_gd(u, y, m=10.0, b=0.0, alpha=0.1, epochs=EPOCHS_A)
loss_a = [h[2] for h in hist_a]
slope_raw, intercept_raw = m / x_std, b - m * x_mean / x_std
print(f"\nPart A, single neuron after {EPOCHS_A} epochs (all {N} points):")
print(f"  y = {slope_raw:.3f} x + {intercept_raw:.2f}   final MSE = {loss_a[-1]:.1f}")
print("  -> basically a flat line at the average y. One neuron can't bend.")

xs = np.linspace(x.min(), x.max(), 300)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.scatter(x, y, s=18, alpha=0.7, label="V data (Anna's Code)")
ax1.plot(xs, slope_raw * xs + intercept_raw, "r-", lw=2.5, label="single-neuron fit")
ax1.set(title="Part A: single neuron, y = m x + b", xlabel="x", ylabel="y")
ax1.legend()
ax2.plot(range(1, EPOCHS_A + 1), loss_a, color="tab:orange", lw=2)
ax2.set(title="Part A: loss vs epoch", xlabel="epoch", ylabel="MSE")
ax2.annotate("levels off well above 0:\na straight line can't fit a V",
             xy=(EPOCHS_A * 0.6, loss_a[-1]), xytext=(EPOCHS_A * 0.35, max(loss_a) * 0.55),
             arrowprops=dict(arrowstyle="->"))
fig.tight_layout()
fig.savefig(HERE / "single_neuron.png", dpi=150)


# ================================================================ Part B
# Normalize y too for the network (Anna's Code does this). We convert back for plotting.
y_mean, y_std = y.mean(), y.std()
U = torch.tensor(u, dtype=torch.float32).unsqueeze(1)
Yn = torch.tensor((y - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)


class VNet(nn.Module):
    """1 input -> hidden layer of 2 ReLU neurons -> 1 linear output neuron."""

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(1, 2)   # n1 = w1*u + c1,  n2 = w2*u + c2
        self.relu = nn.ReLU()           # bends each line: max(0, n)
        self.out = nn.Linear(2, 1)      # y_hat = v1*ReLU(n1) + v2*ReLU(n2) + d

    def forward(self, u):
        return self.out(self.relu(self.hidden(u)))


def train(seed, epochs=400, lr=0.1):
    torch.manual_seed(seed)                        # like "Randomize weights"
    net = VNet()
    opt = torch.optim.SGD(net.parameters(), lr=lr) # plain gradient descent
    loss_fn = nn.MSELoss()
    losses = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(net(U), Yn)   # forward pass + MSE
        loss.backward()              # backpropagation: chain rule through ReLU
        opt.step()                   # w = w - lr * dL/dw for every weight
        losses.append(loss.item())
    return net, losses


# Sometimes a random start leaves a ReLU neuron "dead" (off for every point, so it never
# learns). That's the same as hitting Randomize again on Anna's site: try a new seed.
for seed in range(20):
    net, loss_b = train(seed)
    if loss_b[-1] < 0.02:
        break
print(f"\nPart B, 2-ReLU network (seed {seed}, {len(loss_b)} epochs): "
      f"final normalized MSE = {loss_b[-1]:.4f}  (= {loss_b[-1] * y_std**2:.1f} in raw units)")

# Convert the weights back to raw x and y units so you can compare with Anna's site.
w = net.hidden.weight.detach().numpy().ravel()
c = net.hidden.bias.detach().numpy()
v = net.out.weight.detach().numpy().ravel()
d = net.out.bias.item()
w_raw = w / x_std
c_raw = c - w * x_mean / x_std
v_raw = v * y_std
d_raw = d * y_std + y_mean
print("  Hidden layer (raw x units):")
for i in range(2):
    turn = -c_raw[i] / w_raw[i]
    side = "on for x >" if w_raw[i] > 0 else "on for x <"
    print(f"    n{i+1} = ReLU({w_raw[i]:.3f} x + {c_raw[i]:.2f})   {side} {turn:.1f}")
print(f"  Output: y = {v_raw[0]:.2f} n1 + {v_raw[1]:.2f} n2 + {d_raw:.2f}")
print(f"  Arm slopes: {w_raw[0]*v_raw[0]:+.2f} and {w_raw[1]*v_raw[1]:+.2f}  "
      "(the true V, y = 2|x - 60|, has slopes +2 and -2)")

with torch.no_grad():
    us = torch.tensor((xs - x_mean) / x_std, dtype=torch.float32).unsqueeze(1)
    fit = net(us).numpy().ravel() * y_std + y_mean
    h = net.relu(net.hidden(us)).numpy()
arm1 = v_raw[0] * h[:, 0]
arm2 = v_raw[1] * h[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.scatter(x, y, s=18, alpha=0.7, label="V data (Anna's Code)")
ax1.plot(xs, fit, "g-", lw=2.5, label="2-ReLU network fit")
ax1.plot(xs, arm1 + d_raw, "--", lw=1.2, color="tab:purple", label="neuron 1 ramp (+ bias)")
ax1.plot(xs, arm2 + d_raw, "--", lw=1.2, color="tab:brown", label="neuron 2 ramp (+ bias)")
ax1.set(title="Part B: hidden layer of 2 ReLU neurons", xlabel="x", ylabel="y")
ax1.legend(fontsize=8)
ax2.plot(range(1, len(loss_b) + 1), loss_b, color="tab:orange", lw=2)
ax2.set(title="Part B: loss vs epoch", xlabel="epoch", ylabel="MSE (normalized)")
fig.tight_layout()
fig.savefig(HERE / "relu_network.png", dpi=150)

print(f"\nSaved {HERE / 'single_neuron.png'} and {HERE / 'relu_network.png'}")
plt.show()
