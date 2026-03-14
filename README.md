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
🔷 Core CLI usage (from project root where `cli_train.py` lives)
```bash
python -m cli_train
```

🔷 Optional CLI flags (✅ fix whatever is causing training results to differ from demo.ipynb ✅)
```bash
python -m cli_train --n_modes 3 --hidden 64 --epochs 6000 --lr 5e-3 --n_points 256 --device cpu --seed 27 --log_every 800
```


---

## Repo Structure

```
inverse-piml-schrodinger/
├── README.md
├── requirements.txt
├── pyproject.toml
├── cli_train.py            # CLI support for generating training data
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
│   ├── README.md         # a blank README.md (✅ need to decide if I want to keep all this ✅)
│   └── references.bib    # sources 
├── assets/
│   ├── images/
│   │   └── interpratibility_axis.png    
└── artifacts/
    ├── project_1_followup.md            # conceptual notes and reflection
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
## 🧜‍♀️ PIML Architecture
```mermaid
%%====================================================================
%%  2️⃣ Project 2 Architecture
%%====================================================================
%%{ init: {
        "theme": "base",
        "themeVariables": {
            "background": "`#0D1117`",
            "lineColor": "`#14B5FF`",
            "fontFamily": "'Aclonica', sans-serif",
            "borderRadius": "16",
        },
        "handDrawn": true,
    }
}%%
%%====================================================================
flowchart TB
    %%-----------------------------------------------------------
    %%  NODE DEFINITIONS
    %%-----------------------------------------------------------
    step0["0️⃣ Define TISE dynamics"]

    spatial_grid["$$x\in[-5,5]$$"]:::spatial
    observed_data["$$\rho_n^\text{obs}, \, E_n^\text{obs} $$"]

    MLP1["MLP"]:::MLP
    MLP2["MLP"]:::MLP
    MLP3["MLP"]:::MLP
    MLP4["MLP"]:::MLP

    potential(("$$V_\theta(x)$$"))
    wavefunction_0(("$$\psi_0^\theta(x)$$")):::eigenfunctions
    wavefunction_1(("$$\psi_1^\theta(x)$$")):::eigenfunctions
    wavefunction_2(("$$\psi_2^\theta(x)$$")):::eigenfunctions

    %%-----------------------------------------------------------
    %% CORE PIPELINE (PIML Framework)
    %%--------------------------------------------------------------
    subgraph PIML["PIML Framework"]
        direction LR

        subgraph synthetic["1️⃣ Synthetic Data"]
            direction TB
            spatial_grid
            observed_data
        end

        subgraph learned_parameters["2️⃣ Learned Parameters"]
            subgraph PINN[" "]
                direction TB

                subgraph potential_network["Potential Network"]
                direction LR
                    MLP1 --> potential
                end

                subgraph eigenfunction_networks["Eigenfunction Networks"]
                    direction TB
                    MLP2 --> wavefunction_0
                    MLP3 --> wavefunction_1
                    MLP4 --> wavefunction_2
                end

            end

            subgraph energy_eigenvalues["Energy Eigenvalues"]
                direction LR
                energy_0(("$$E_1^\theta$$")):::energy
                energy_1(("$$E_1^\theta$$")):::energy
                energy_2(("$$E_2^\theta$$")):::energy
            end

        end
        learned_parameters:::PINN

        spatial_grid --> MLP1
        spatial_grid --> MLP2
        spatial_grid --> MLP3
        spatial_grid --> MLP4

        subgraph finite_difference["3️⃣ Finite Difference"]
            direction LR
            potential_stencil["$$V_\theta''(x)\,$$ via stencil"]
            eigenfunction_stencil["$$\frac{\partial^2}{\partial x^2}\psi_n^\theta(x) \,$$via stencil"]
        end

        subgraph residual["4️⃣ Physics Residual"]
           A["$$R_n(x)=-\frac{1}{2}\frac{\partial^2}{\partial x^2}\psi_n^\theta + V_\theta\psi_n^\theta - E^\theta_n \psi$$"]
        end

        subgraph loss_terms["5️⃣ Loss Function"]
            direction LR
            loss_physics["$$\mathcal{L}_\text{TISE}$$"]
            loss_data["$$\mathcal{L}_\text{data}$$"]
            loss_smooth["$$\mathcal{L}_\text{smooth}$$"]
            loss_norm["$$\mathcal{L}_\text{norm}$$"]
        end
        loss_terms:::loss
        
        optimization["6️⃣ Optimizer (Adam)"]

        observed_data --> loss_data

        wavefunction_0 --> eigenfunction_stencil
        wavefunction_1 --> eigenfunction_stencil
        wavefunction_2 --> eigenfunction_stencil

        eigenfunction_stencil --> residual --> loss_physics

        wavefunction_0 --> loss_norm
        wavefunction_1 --> loss_norm
        wavefunction_2 --> loss_norm

        energy_0 --> loss_data
        energy_1 --> loss_data
        energy_2 --> loss_data

        energy_0 --> residual
        energy_1 --> residual
        energy_2 --> residual

        potential --> residual
        potential --> potential_stencil --> loss_smooth

        loss_terms --> optimization


        optimization -- "training loop" --> MLP1
        optimization -- "training loop" --> MLP2
        optimization -- "training loop" --> MLP3
        optimization -- "training loop" --> MLP4
        optimization -- "training loop" --> energy_0
        optimization -- "training loop" --> energy_1
        optimization -- "training loop" --> energy_2

    end

    %%-----------------------------------------------------------
    %%  External connections
    %%-----------------------------------------------------------
    step0 --> PIML
    
    %%--------------------------------------------------------------
    %%  POST‑TRAINING DIAGNOSTICS
    %%--------------------------------------------------------------
    sanity_checks["6️⃣ Sanity Checks"]:::stage6

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
    POD:::dashed_orange_1

    PIML --> sanity_checks
    PIML --> POD

    %%-----------------------------------------------------------
%% CORE STAGES (pseudo gradient)
%%-----------------------------------------------------------

style step0 fill:#0f0a25,stroke:#7c5cff,stroke-width:4px,color:#ffffff,rx:12px, ry:12px;

style PIML fill:#0b1020,stroke:#5a6cff,stroke-width:4px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;

style synthetic fill:#14153a,stroke:#6b4cff,stroke-width:3px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;
style spatial_grid fill:#1b133d,stroke:#9a66ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;
style observed_data fill:#181c3f,stroke:#5f88ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

%%-----------------------------------------------------------
%% NETWORKS
%%-----------------------------------------------------------

classDef MLP fill:#2a185c,stroke:#9a66ff,stroke-width:4px,color:#ffffff,rx:12px, ry:12px;
style eigenfunction_networks fill:#11253f,stroke:#38c6ff,stroke-width:2px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;
style learned_parameters fill:#0d1a30,stroke:#3b82ff,stroke-width:3px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;
style PINN fill:#131130,stroke:#7259ff,stroke-width:3px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;

classDef eigenfunctions fill:#153a55,stroke:#4ec9ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

style potential fill:#11163a,stroke:#5f88ff,stroke-width:4px,color:#ffffff,rx:12px, ry:12px;
style potential_network fill:#0d1630,stroke:#3072f5,stroke-width:4px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;

classDef energy fill:#0d2238,stroke:#3b9eff,stroke-width:4px,color:#ffffff,rx:12px, ry:12px;
style energy_eigenvalues fill:#071320,stroke:#3b9eff,stroke-width:4px,color:#ffffff,stroke-dasharray: 6 6, rx:12px, ry:12px;

%%-----------------------------------------------------------
%% PHYSICS OPERATORS
%%-----------------------------------------------------------

style finite_difference fill:#0a2330,stroke:#63e2ff,stroke-width:2px,color:#ffffff,stroke-dasharray: 6 6,rx:12px, ry:12px;
style potential_stencil fill:#112042,stroke:#5f88ff,stroke-width:3px,color:#ffffff,rx:12px, ry:12px;
style eigenfunction_stencil fill:#132a3a,stroke:#4ec9ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

style residual fill:#0a1329,stroke:#5f88ff,stroke-width:2px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;
style A fill:#0e1c3f,stroke:#4ec9ff,stroke-width:2px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;

%%-----------------------------------------------------------
%% LOSS (cyan end of gradient)
%%-----------------------------------------------------------

classDef loss fill:#0a0f1a,stroke:#22d3ee,stroke-width:2px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;
style loss_terms fill:#0b1326,stroke:#22d3ee,stroke-width:2px,color:#ffffff,stroke-dasharray:6 6,rx:12px, ry:12px;

style loss_physics fill:#0d1e3b,stroke:#38c6ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;
style loss_norm fill:#0e2a3b,stroke:#4ec9ff,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;
style loss_smooth fill:#0b2f2a,stroke:#34d399,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;
style loss_data fill:#1e153a,stroke:#c084fc,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

%%-----------------------------------------------------------
%% OPTIMIZER (accent color)
%%-----------------------------------------------------------

style optimization fill:#2a1c16,stroke:#f59e0b,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

%%-----------------------------------------------------------
%% POST-TRAINING DIAGNOSTICS
%%-----------------------------------------------------------

classDef stage6 fill:#2a1a1a,stroke:#f87171,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;
classDef stage7 fill:#2a1a1a,stroke:#fb7185,stroke-width:2px,color:#ffffff,rx:12px, ry:12px;

classDef dashed_orange_1 fill:#161b22,stroke:#fb7185,stroke-dasharray:6 6,color:#ffffff,rx:12px, ry:12px;

%%-----------------------------------------------------------
%% LINKS
%%-----------------------------------------------------------

linkStyle default stroke:#8b949e,stroke-width:2px;
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
##### 🧩 Mathematical Formulation

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

Taking the SVD of $\mathbf{\Psi^\theta}$ gives 

$$\mathbf{\Psi^\theta}=U\Sigma W^T$$

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
