# inverse-pinn-schrodinger

> 🥅 **Goal:** understand which operator features are robustly recoverable under strong physics priors and limited data.


This project extends the `lf-pinn-harmonic-oscillator` framework to an inverse 1D Schrödinger problem.

While project 1 investigated the robustness  of Physics-Informed Neural Networks (PINNs) under low fidelity discretization for a *known Hamiltonian*, this project investigates what information about an *unknown potential* that can be recovered from partial, noisy observations of quantum states.

Using the time-independent Schrödinger equation (TISE) as a physics constraint, we treat:
- The potential $V(x)$ as a learnable function 
- The wavefunctions act as auxiliary fields (❓❓what do we mean by auxiliary fields in this context?❓❓) constrained by the PDE 
- The eigenvalues $E_n$ as trainable scalars.

The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

> ✨  As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability.

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
Core CLI usage (from project root where `cli_train.py` lives)
```bash
python -m cli_train
```

🔵 **Example:** Optional CLI flags ✅Make sure this is accurate!✅
```bash
python -m cli_train \
  --n_modes 3 \
  --hidden 64 \
  --epochs 4000 \ 
  --lr 5e-3 \ 
  --n_points 256 \
  --device cuda \ 
  --seed 27 \
  --log_every 500
```

---

## Repo Structure

```
inverse-piml-schrodinger/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── model.py            # neural network ansatz for V_theta(x)
│   ├── physics.py          # TISE residual, data mismatch, scale-aware smoothness
│   ├── inverse.py          # inverse-specific losses
│   ├── pod.py              # POD decomposition + reconstruction
│   ├── train.py            # training loop and CLI
│   ├── visualizations.py   # visualization utilities (e.g., seeding and grids)
│   └── utils.py            # helper functions (e.g., seeding and grids)
├── notebooks/
│   └── demo.ipynb        # visual + narrative
├── references/
│   ├── README.md         # a blank README.md idk what for though ❓
│   └── references.bib    # sources 
├── assets/
│   ├── images/
│   │   └── interpratibility_axis.png    
└── artifacts/
    ├── notes.md                         # conceptual notes and reflection
    ├── figures.md                       # structure-based analysis of demo visuals
    └── demo_visuals/    
        ├── density.png
        ├── learned_energies.png
        ├── learned_potential.png
        ├── learned_wavefunctions.png
        ├── overlap_heatmap.png
        ├── pod_modes.png
        ├── pod_singular_values.png
        └── training_curves.png                       
```

---
## 🏗️ PIML Architecture
```mermaid
%%====================================================================
%%  CURVED‑CORNER MERMAID – DARK THEME + PURPLE ACCENT
%%====================================================================
%%{ init: {
        "theme": "base",
        "themeVariables": {
            "background": "#0d1117",
            "lineColor": "#14b5ff",
            "textColor": "#ffffff",
            "fontFamily": "'Aclonica', sans-serif",
            "borderRadius": "16",
        },
        "handDrawn": true
    } }%%
%%====================================================================

flowchart TB
    %%--------------------------------------------------------------
    %%  STYLE DEFINITIONS – keep original dark shades, add purple
    %%--------------------------------------------------------------
    classDef stage0 fill:#301934,stroke:#5D3FD3,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% deep purple for physics 
    classDef stage1 fill:#0f2a3d,stroke:#14b5ff,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% original dark indigo 
    classDef stage2 fill:#103b4f,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% original teal blue 
    classDef stage3 fill:#124f55,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% original cyan/green 
    classDef stage4 fill:#1a6b63,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% original orangeish teal 
    classDef stage5 fill:#1f4e5f,stroke:#f78166,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% optimizer 
    classDef stage6 fill:#3a2f2a,stroke:#f78166,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% diagnostics
    classDef stage7 fill:#0f2a3d,stroke:#14b5ff,stroke-width:2px,color:#ffffff,rx:12,ry:12;   %% notes / tiny bubbles 

    classDef PIML_framework fill:#161b22,stroke:#14b5ff,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef Observed_data stroke:#0f2a3d,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef Total_loss stroke:#00f5db,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef POD_diagnostics stroke:#f78166,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;

    %%--------------------------------------------------------------
    %%  LIGHT‑GRAY ARROW STYLE (so arrows stay subtle)
    %%--------------------------------------------------------------
    linkStyle default stroke:#888,stroke-width:2px

    %%--------------------------------------------------------------
    %%  CORE PIML PIPELINE
    %%--------------------------------------------------------------
    subgraph PIML["PIML Framework"]
        direction TB

        subgraph synthetic_data["1️⃣ Synthetic Data"]
            direction TB
            B["Spatial grid"]:::stage1
            obs["Noisy density & energy samples"]:::stage1
        end
        synthetic_data:::Observed_data

        C["2️⃣ Neural Ansatz\nMLPs (potential function and \n wavefunction-energy eigenvalue pairs)"]:::stage2

        D["3️⃣ Automatic Differentiation"]:::stage3

        subgraph loss["4️⃣ Total Loss"]
            direction TB
            physics["- Physics residual loss\n- Wavefunction‑norm loss\n- Smoothness regularizer"]:::stage4
            data["Data‑misfit loss (density & energy)"]:::stage4
        end
        loss:::Total_loss

        F["5️⃣ Optimizer (Adam)\nupdates all MLP weights"]:::stage5

        %% Connections (light‑gray arrows)
        B --> C
        obs --> data
        C --> D
        D --> loss
        loss --> F
        F -- training loop --> C
    end

    %%--------------------------------------------------------------
    %%  POST‑TRAINING DIAGNOSTICS
    %%--------------------------------------------------------------
    H["6️⃣ Sanity Checks"]:::stage6

    subgraph POD["7️⃣ POD Diagnostics"]
        direction TB
        SVD["SVD of snapshot matrix"]:::stage6
        sigma["POD spectrum (i.e., singular values)"]:::stage6
        U["POD eigenmodes (columns of U)"]:::stage6
        %% noteW["Wᵀ holds temporal coefficients – not needed for static eigenmode analysis"]:::stage7
        %% noteSpec["‘Spectrum’ = the set of singular values; gaps signal low‑rank structure"]:::stage7

        SVD --> sigma
        SVD --> U
    end
    
    A["0️⃣ Define TISE dynamics"]:::stage0

    %%--------------------------------------------------------------
    %%  OUTER LINKS
    %%--------------------------------------------------------------
    A --> PIML:::PIML_framework
    PIML --> H
    PIML -- snapshot matrix --> POD:::POD_diagnostics
```

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

#### 3️⃣ Automatic Differentiation - Recover the first and second partial deriviatives of the learned wavefunctions with respect to $x$
$$\frac{\partial}{\partial x}\psi_n^\theta$$
$$\frac{\partial^2}{\partial x^2}\psi_n^\theta$$

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

Taking the SVD of $\mathbf{\Psi^\theta}$ gives $$\mathbf{\Psi^\theta}=U\Sigma W^T$$
where 
- the columns of $U$ are POD spatial modes of the learned eigenfunctions.
- the diagonal elements of $\Sigma$ are the corresponding singular values.
- $W^T$ contains temporal coefficients (not needed for TISE).
> 📝 The spectrum reveals learned structure where large gaps in consecutive singular values indicate low-rank structure is learned correctly. 

> ✨ Importantly, POD does not enforce physics. It reveals structure. This is important for interpretability!
---
## 🌍 Global Design Choices
Assumptions:
- Atomic units: $\hbar = 1$
- Normalized parameters: $m = 1$ electron rest mass
- Wavefunction/eigenmode normalization is handled either implicitly or by the PDE
- probability density observations are on an absolute scale

What is being learned:
- The potential function $V_\theta(x) := V(\theta, x)$ via the MLP.
- Eigenmodes $\psi_n^\theta(x) := \psi_n(\theta,x)$ via the MLP. 
- Associated energy eigenvalues $E_n^\theta := E_n(\theta)$ as learnable scalars. 

Orthogonality
- For our low fidelity design, we are not _enforcing_ orthogonality directly.
- However, our POD function (`src/pod.py`) allows us to _diagnose_ orthogonality.

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
