# Spring-Mass PINN Simulator

A Physics-Informed Neural Network (PINN) simulator for the classic spring-mass system — a mass *m* suspended from a ceiling by a spring of stiffness *k*. The neural network learns to satisfy the governing ODE directly, without any labeled training data, and the result is visualized in an interactive GUI with real-time animation.

---

## Physics Background

The displacement *x(t)* from the equilibrium position satisfies:

```
m·x''(t) + k·x(t) = 0
```

with initial conditions:

```
x(0)  = x₀   (initial displacement)
x'(0) = v₀   (initial velocity)
```

The analytical solution is:

```
x(t) = x₀·cos(ωt) + (v₀/ω)·sin(ωt),   ω = √(k/m)
```

The PINN is trained to recover this solution using only the ODE and initial conditions — no ground-truth data is used.

---

## How the PINN Works

| Component | Detail |
|-----------|--------|
| Architecture | Fully-connected network, 5 hidden layers × 64 neurons, Tanh activation |
| Input | Scalar time `t ∈ [0, T]` |
| Output | Scalar displacement `x(t)` |
| Physics loss | Mean squared ODE residual over 300 collocation points |
| IC loss | Squared error on `x(0)` and `x'(0)`, weighted ×100 |
| Total loss | `L = L_physics + 100 · L_IC` |
| Optimizer | Adam (lr = 1e-3) with Cosine Annealing LR scheduler |

Derivatives are computed via **automatic differentiation** (`torch.autograd.grad`) so the network is penalized for violating the equation of motion at every collocation point.

---

## Features

- **Interactive parameter sliders** — adjust *m*, *k*, *x₀*, *v₀*, and simulation duration *T* before training
- **Non-blocking training** — runs in a background thread; the GUI stays responsive with a live progress bar and loss readout
- **Four live plots** updated after training:
  - Spring-mass animation (synchronized with the position plot)
  - Position vs. time — PINN vs. analytical solution
  - Phase-space portrait (*x* vs. *v*)
  - Training loss curve (log scale)
- **Real-time animation** — spring stretches and compresses as the mass oscillates; a time marker sweeps across the position plot simultaneously

---

## Requirements

| Package | Version |
|---------|---------|
| Python  | ≥ 3.8   |
| PyTorch | ≥ 1.10  |
| NumPy   | ≥ 1.21  |
| Matplotlib | ≥ 3.5 |
| Tkinter | (stdlib, included with standard Python on macOS/Linux) |

Install dependencies:

```bash
pip install torch numpy matplotlib
```

---

## Usage

```bash
python spring_mass_pinn.py
```

### Workflow

1. **Set parameters** using the sliders in the left panel.
2. Click **▶ Train PINN** — training runs in the background (progress bar updates live).
3. Once training completes, the four plots are populated automatically.
4. Click **▷ Start Animation** to watch the spring-mass motion driven by the PINN prediction.
5. Click **↺ Reset** to clear results and start over with new parameters.

---

## GUI Layout

```
┌─────────────────┬────────────────────────────────────────────┐
│  Parameters     │  Spring-Mass Animation  │  Position vs Time │
│  m  [slider]    ├─────────────────────────┼───────────────────┤
│  k  [slider]    │  Phase Space            │  Training Loss    │
│  x₀ [slider]   │                         │                   │
│  v₀ [slider]   └─────────────────────────┴───────────────────┘
│  T  [slider]
│  Epochs [spin]
│  ─────────────
│  ▶ Train PINN
│  ▷ Start Animation
│  ↺ Reset
│  ─────────────
│  Status / Progress / Loss
└─────────────────
```

---

## File Structure

```
03015_pinn_mk/
├── spring_mass_pinn.py   # Main application (PINN model + training + GUI)
└── README.md             # This file
```

---

## Example Results

With default parameters (`m = 1 kg`, `k = 4 N/m`, `x₀ = 1 m`, `v₀ = 0 m/s`, `T = 10 s`, `epochs = 3000`):

- Angular frequency: `ω = √(k/m) = 2 rad/s`
- Period: `T = π ≈ 3.14 s`
- The PINN solution closely matches the analytical cosine curve after training, and the phase-space portrait converges to a near-perfect ellipse.

Increasing epochs (e.g., 10 000–20 000) further reduces the loss and improves accuracy for longer time horizons.
