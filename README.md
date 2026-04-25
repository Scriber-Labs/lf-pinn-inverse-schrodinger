# inverse-pinn-schrodinger

> 🥅 **Goal:** understand which operator features are robustly recoverable under strong physics priors and limited data.


This project extends the `lf-pinn-harmonic-oscillator` framework to an inverse 1D Schrödinger problem.

While [project 1](https://github.com/Scriber-Labs/lf-pinn-harmonic-oscillator) investigated the robustness of Physics-Informed Neural Networks (PINNs) under low fidelity discretization for a *known Hamiltonian*, this project investigates what information about an *unknown potential* that can be recovered from partial, noisy observations of quantum states.

Using the time-independent Schrödinger equation (TISE) as a physics constraint, we treat:
- The potential $V(x)$ as a learnable function. 
- The wavefunctions $\psi_n(x)$ as auxiliary fields constrained by the PDE.
- The eigenvalues $E_n$ as trainable scalars.

The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

> 🎗️ As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability.

---
## 🔰 Quick Start
```bash
# Install dependencies
pip install -e .

# Train a model with default settings
python -m train

# Launch demo notebook
jupyter lab notebooks/demo.ipynb
```
🔷 Core CLI usage (from project root where `cli_train.py` lives)
```bash
python -m cli_train
```

🔷 Optional CLI flags (✅ fix whatever is causing training results to differ from demo.ipynb ✅)
```bash
!python3 -m cli.cli_train \
    --lambda-data 1.5 \
    --lambda-physics 0.5 \
    --lambda-smooth 0.25 \
    --lambda-ordered 1.0
```


---

## Repo Structure
```text
lf-pinn-inverse-schrodinger/
├── artifacts/
│   ├── demo_visuals/
│   │   ├── cross_overlap_heatmap.png
│   │   ├── density.png
│   │   ├── learned_energies.png
│   │   ├── learned_potential.png
│   │   ├── learned_wavefunctions.png
│   │   ├── overlap_heatmap.png
│   │   ├── pod_modes.png
│   │   ├── pod_singular_values.png
│   │   └── training_curves.png
│   ├── figures.md
│   ├── interpretability.md
│   └── project_1_followup.md
├── assets/
│   ├── images/
│   │   ├── loss_table.png
│   │   ├── loss_table_extended.png
│   │   ├── loss_table_numbered.png
│   │   └── project_2_architecture.png
│   ├── mermaid-diagrams/
│       ├── architecture.mmd
│       ├── blue_to_orange_gradient.mmd
│       ├── eigenscribe-theme_gradient.mmd
│       ├── purple_gradient.mmd
│       └── vanilla_architecture.mmd
├── docs/
│   ├── architecture.pdf
│   └── loss function table (extended).pdf
├── notebooks/
│   └── demo.ipynb
├── src/
│   ├── __init__.py
│   ├── inverse.py
│   ├── model.py
│   ├── physics.py
│   ├── pod.py
│   ├── train.py
│   ├── utils.py
│   └── visualizations.py
├── CITATION.cff
├── LICENSE
├── README.md
├── cli_train.py
├── pyproject.toml
├── references.bib
├── repo_summary.py
└── requirements.txt

```

---
## 🌍 Global Design Choices
### Assumptions
- Atomic units: $\hbar = 1$
- Normalized parameters: $m = 1$ electron rest mass
- Wavefunction/eigenmode normalization is handled either implicitly or by the PDE
- probability density observations are on an absolute scale

### What is being learned
- The potential function $V_\theta(x) := V(\theta; x)$ via the MLP.
- Eigenmodes $\psi_n^\theta(x) := \psi_n(\theta;x)$ via the MLP. 
- Associated energy eigenvalues $E_n^\theta := E_n(\theta)$ as learnable scalars. 

### Orthogonality 
- For this low fidelity implementation, we are not _enforcing_ orthogonality directly.
- However, our POD function (`src/pod.py`) allows us to _diagnose_ orthogonality.

---
## 🧜‍♀️ PIML Architecture
![Project 2 Architecture](assets/images/project_2_architecture.png)

### 🗺️ Mathematical Mapping for PIML Architecture
#### 0️⃣ Problem Setup - Initialize the 1D time-independent Schrödinger equation (TISE) with random values and weights
##### 🧩 Mathematical Formulation
$$-\frac{1}{2}\frac{d^2}{dx^2}\psi^\theta_n(x) + V_\theta(x)\psi_n^\theta(x)=E_n^\theta\psi_n^\theta(x), \quad x\in[-5,5]$$

> 🥅 The overall goal is to learn an unknown ground-truth potential $V(x)$ and wavefunctions $\psi_n$ along with their associated energy eigenvalues $E_n$ from noisy, low-fidelity data.

#### 1️⃣ Synthetic Data - Physics enforcement via noisy observations
##### 🧩 Mathematical Formulation
###### Deterministic, uniformly spaced 1D grid of spatial points:
$$x_i=x_0+i\Delta x  \quad \forall i\in\{ 0,1,2,\dots, N-1 \}$$ where $$\Delta x=\frac{x_{N-1}-x_0}{N-1} \ .$$

###### Noisy observations
1. **Observed probability densities:**

$$\rho_n(x_i)^\text{obs}=|\psi_n^\text{true}(x_i)|^2+\sigma_\rho\,\mathcal{N}(0,1),  \qquad  \sigma_\rho = 0.02$$

2. **Observed energies:**

$$E_n^\text{obs}=E_n^\text{true}+\sigma_E\,\mathcal{N}(0,1), \qquad \sigma_E=0.05$$

> 📝 Noisy observations simulate sparse experimental observations.

#### 2️⃣ Neural Ansatz - Learned potential and eigenstates
##### 🧩 Mathematical Formulation:
$$V_\theta(x)=\text{MLP}_V(\theta_V;x)$$
$$\psi_n^\theta=\text{MLP}_\psi(\theta_\psi;x)$$
$$E_n^\theta=\text{learnable scalar}$$

#### 3️⃣ Finite Difference



#### 4️⃣ Loss Function- Low fidelity PINN objective function with four loss terms
##### 🧩 Mathematical Formulation
**Individual terms:**

<p align="center">
  <img src="./assets/images/loss_table_numbered.png"
       alt="Loss table."
       height="250">
</p>

**Composite Objective:** 

$$\mathcal{L}_\text{total} = \lambda_\text{TISE}\mathcal{L}_\text{TISE} + \lambda_\text{norm}\mathcal{L}_\text{norm} + \lambda_\text{smooth}\mathcal{L}_\text{smooth} + \lambda_\text{data}\mathcal{L}_\text{data}$$


#### 5️⃣ Optimization - Gradient descent update
##### 🧩 Mathematical Formulation:
$$\theta \leftarrow \theta - \eta\nabla_\theta\mathcal{L}_\theta$$

#### 6️⃣ Sanity Checks - Validate the following:
- Orthogonality of learned eigenfunctions
- Energy ordering $E_0^\theta < E_1^\theta < E_2^\theta$
- Smoothness of learned potential
- Boundary decay behavior

> 📝 Sanity checks via figure analysis are an essential component of the sanity check process and for providing interpretable insights (see `./artifacts/figures.md`).

#### 7️⃣ Proper Orthogonal Decomposition (POD) Diagnostics - Take the SVD of the snapshot matrix for learned wavefunctions
##### 🧩 Mathematical Formulation:
$$\mathbf{\Psi^\theta}=[\psi_0^\theta, \dots, \psi_{N-1}^\theta] \quad \text{(snapshot matrix)}$$

where each column corresponds to a learned wavefunction for each spatial grid point for the $N-1^\text{th}$ eigenmode.

Taking the SVD of $\mathbf{\Psi^\theta}$ gives 

$$\mathbf{\Psi^\theta}=U\Sigma W^T$$

where 
- the columns of $U$ are POD spatial modes of the learned eigenfunctions.
- the diagonal elements of $\Sigma$ are the corresponding singular values.
- $W^T$ contains temporal coefficients (not needed for TISE).
> 📝 The spectrum reveals learned structure where large gaps in consecutive singular values indicate low-rank structure is learned correctly. 

> ✨ Importantly, POD does not enforce physics. It reveals structure. This is important for interpretability!

---
## 🔮 Future possible implementations
- stochastic spatial grid sampling ($x\sim\mathcal{U}(x_\text{min},x_\text{max})$ instead of deterministic, equally-spaced spatial grid)
- unnormalized densities
- increase the number of learned eigenmodes
- partial observation windows
- Additional loss terms:
  - orthogonality constraints between $\psi_n(\theta,x)$ 
  - add energy ordering regularization
  - add symplectic loss
  - Dynamic weighting of the loss terms
    - 📝 For this project, the loss weights are assigned to a static `dict`. 
    - Plan of Action
      - Focus on data first, then enforce physics loss.
      - Prevent over-smoothing early during training.
      - **Adaptive balancing:** Compute the magnitude of each loss term every epoch and scale the $\lambda$'s so that all terms contribute roughly the same amount.
      - **Bayesian/ probabilistic sampling:** Treat each $\lambda$ as a learnable hyperparameter and update it with gradient descent.
    - Tips
      - Start with a static `dict` for the first pass. This will provide a baseline to compare against.
      - Add a **scheduler** only if you see a failure mode symptom.
    
        | **Symptom** | **Remedy**                                                       |
        | ----------- |------------------------------------------------------------------|
        | Physics residual stalls while the data-fit continues to improve. | Increase $\lambda_\text{physics}$.                               |
        | Flattening failure mode | Decrease $\lambda_\text{smooth}$ or slow down its schedule.      |
         | Wildly oscillating loss curves | Use a **smooth ramp** for all $\lambda$'s to stabilize training. | 
    
      - Log the effective $\lambda$ values each epoch and plot them alongside the training curves. This will help reveal whether a scheduler helped or hindered learning.
- use PySR architecture instead of 'pure' neural network (Crammer, 2023)
  - ✨ adds to interpretability discussion in `lf-pinn-inversse-schrodinger`

---
## ✅ To Do
- [ ] Figure 8 (bifurcation diagram) in `demo.ipynb`
- [ ] `figures.md`
  - Each figure should have:
    - Title
    - What is shown
    - What it tells us
    - Possible failure modes
    
    🔵 Example tone:
      > *"Figure 5 reveals partial orthogonality between learned eigenfunctions, despite no explicit orthogonality constraint. This suggests that the TISE residual alone imposes meaningful structure, though small off-diagonal overlaps indicate residual mode mixing."*
---
