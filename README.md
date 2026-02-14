# inverse-piml-schrodinger

> 🥅 **Goal:** understand which operator features are robustly recoverable under strong physics priors and limited data.


This project extends the `lf-pinn-harmonic-oscillator` framework to an inverse quantum problem.
While project 1 investigated the robustness  of physics-informed neural networks (PINNs) under low fidelity discretization for a *known Hamiltonian*, this project is concerned with the information about an *unknown potential* that can be recovered from partial, noisy observations of quantum states.

Using the time-independent Schrodinger equation (TISE) as a physics constraint, we treat the potential $V(x)$ as a learnable function while wavefunctions act as auxiliary fields constrained by the PDE. The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability.

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
│   ├── README.md         # a blank README.md idk what for though. chat gpt told me to include to when i decided to make the references folder
│   └── references.bib    # sources (❓🙋🏻‍♀️ not necessarily used explicity by the github repo code, but that's mainly because i have no idea how to do that. if there is a standard way of implementing bib files easily like in overleaf im open to something like that... just no relearning entire languages and new software related bullshit (pardon my French)
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
%%  CURVED-CORNER MERMAID WITH SUBGRAPH HEADER
%%====================================================================
%%{ init: {
        "theme": "base",
        "themeVariables": {
            "background": "#0d1117",
            "lineColor": "#14b5ff",
            "textColor": "#ffffff",
            "fontFamily": "'Aclonica', sans-serif",
            "borderRadius": "16"       /* larger radius for more rounded corners */
        },
        "handDrawn": true
    } }%%
%%====================================================================

flowchart TB

    %%--------------------------------------------------------------
    %%  COLOR RAMP (pseudo-gradient)
    %%--------------------------------------------------------------
    classDef stage0 fill:#0b1c2d,stroke:#14b5ff,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage1 fill:#0f2a3d,stroke:#14b5ff,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage2 fill:#103b4f,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage3 fill:#124f55,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage4 fill:#1a6b63,stroke:#00f5db,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage5 fill:#1f4e5f,stroke:#f78166,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage6 fill:#3a2f2a,stroke:#f78166,stroke-width:2px,color:#ffffff,rx:12,ry:12;
    classDef stage7 fill:#0f2a3d,stroke:#14b5ff,stroke-width:2px,color:#ffffff,rx:12,ry:12;

    classDef PIML_framework fill:#161b22,stroke:#14b5ff,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef Observed_data stroke:#0f2a3d,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef Total_loss stroke:#00f5db,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    classDef POD_diagnostics stroke:#f78166,stroke-dasharray:6 6,color:#ffffff,rx:12,ry:12;
    
    %%--------------------------------------------------------------
    %%  MAIN PIPELINE SUBGRAPH WITH HEADER
    %%--------------------------------------------------------------
    subgraph PIML["PIML Framework"]
        direction TB
        subgraph synthetic_data["1️⃣ Synthetic Data"]
            direction TB
            B["Spatial Grid"]:::stage1
            obs["Noisy Observations"]:::stage1
        end
        
        synthetic_data:::Observed_data
        C["2️⃣ Neural Ansatz"]:::stage2
        D["3️⃣ Automatic Differentiation"]:::stage3
        F["5️⃣ Optimizer (Adam)"]:::stage5
        
        subgraph loss["4️⃣ Total Loss"]
            direction TB
            physics["Physics Loss + Wavefunction Normalization Loss + Smoothness Regularization"]:::stage4

            data["Data mismatch loss"]:::stage4
        end
        
        B --> C
        obs --> data
        C --> D
        D --> loss:::Total_loss
        loss --> F
        F -- training loop --> C
    end
    
    %%--------------------------------------------------------------
    %%  CONTEXT & DIAGNOSTICS
    %%--------------------------------------------------------------
    A["0️⃣ Define TISE Dynamics"]:::stage0
    H["6️⃣ Sanity Checks"]:::stage6
    
    subgraph POD["POD Diagnostics"]
        direction TB
        SVD["7️⃣ SVD"]:::stage6
        sigma["8️⃣ POD Singular Values"]:::stage6
        U["9️⃣ POD Eigenmodes"]:::stage6
        
        SVD --> sigma
        SVD --> U
    end
    
    A --> PIML:::PIML_framework
    PIML --> H
    PIML -- Snapshot Matrix --> POD:::POD_diagnostics
```

### Pipeline Legend (Mathematical Mapping)

| Step | Component                 | Mathematical Description                                                                                                                                                | Interpretation                                                                                |
|----|---------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| 0️⃣ | Problem setup             |                                                                                                                                                                         | Define the physical system                                                                    |
| 1️⃣ | Spatial grid              | $x \in [-5, 5]$                                                                                                                                                         | Synthetic “data” for physics enforcement                                                      |
| 2️⃣ | Neural ansatz             | $\begin{align*} V_\theta(x) &= \text{MLP}_V(x; \theta_V) \\ \psi_n^\theta(x) &= \text{MLP}_\psi(x; \theta_\psi) \\ E_n^\theta &= \text{learnable scalar}  \end{align*}$ | Learned potential and eigenstates                                                             |
| 3️⃣ | Automatic differentiation | $\frac{\partial}{\partial x}\psi_n^\theta$ and $\frac{\partial^2}{\partial x^2}\psi_n^\theta$                                                                           | Recover first and second partial derivatives of the learned wavefunctions with respect to $x$ |
| 4️⃣ | Total loss                | $\mathcal{L}_\text{tot}=\mathcal{L}_\text{TISE}+\mathcal{L}_\text{norm}+\mathcal{L}_\text{smooth}+\mathcal{L}_\text{data}$                                              | Low-fidelity PINN objective                                                                   |
| 5️⃣ | Optimization              | $\theta \leftarrow \theta - \eta\nabla_\theta \mathcal{L}$                                                                                                              | Gradient-based learning                                                                       |
| 6️⃣ | Sanity Checks             |                                                                                                                                                                         | Sanity checks and structure validation                                                        |
| 7️⃣ | POD Diagnostics           |                                                                                                                                                                         | Proper Orthogonal Decomposition of the learned eigenfunctions                                 |

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
## 📉 Loss Function

| **Loss Term** | **Formulation**                                                                                                                                                                | **Soft vs. Hard**                        | **Type**              | **Comments** |
| ------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|-----------------------| ------------ |
| **Schrodinger residual (physics loss)** | $$\mathcal{L}_\text{TISE} = \sum_n{\Big\| -\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}\psi_n^\theta + V_\theta(x)\psi_n^\theta(x)-E_n^\theta\psi_n^\theta(x) \Big\|^2}$$ | soft                                     | Consistency condition | Enforces the TISE. |
| **Wavefunction normalization loss** | $$ \mathcal{L}_\text{norm} = \frac{1}{N}\sum_{n=1}^N{\bigg(\int{\|\psi^\theta_n(x)\|^2dx} - 1\bigg)^2}$$                                                                       | soft                                     | Regularizer           | Enforces normalization structure 
| **Scale-aware smoothness** | $$ \mathcal{L}_\text{smooth} = \Big< \frac{\| V_\theta''(x)\| ^2}{\epsilon + \|V_\theta(x)\|^2}\Big> $$                                                                        | soft                                     | Regularizer           | Penalize steep curvature in $V_\theta(x)$. Encourages physically plausible potentials and controls the ill-posedness of the inverse problem. |
| **Data mismatch (obervables)** | $$ \mathcal{L}_\text{data} = \sum_n{\|E_n^\theta-E_n^\text{obs}\|^2 +  \|\|\psi_n^\theta(x)\|^2 - \rho_n^\text{obs}(x)\|^2} $$ | soft (in the sense of measurement noise) | Consistency condition | Ensures the learned eigenstates match noisy observations. |
| **Total Loss** | $$ \mathcal{L}_\text{total} = \lambda_\text{TISE}\mathcal{L}_\text{TISE} + \lambda_\text{norm}\mathcal{L}_\text{norm} + \lambda_\text{smooth}\mathcal{L}_\text{smooth} + \lambda_\text{data}\mathcal{L}_\text{data} $$ | ❓                                        | ❓                     | ❓ |

---
## Proper Orthogonal Decomposition (POD)
> ✨ POD does not enforce physics. It reveals structure.

### Specific Questions POD Answers

| 🧙🏻‍♂️ Question                                          | ✨ Relvance                     |
| -----------------------------------------------------| ------------------------------- |
| Are $\psi_n(\theta,x)$ distinct or collapsing?       | Detects mode collapse           |
| How many effective modes exist?                      | Identifiability                 |
| Are learned states (❓is this just the same thing as saying 'learned eigenmodes') redundant?                     | Overparameterization            |
| Do modes align with energy ordering?                 | Model consistency               |
| Is orthogonality emerging naturally?                 | Strength of physics constraints |

### 🏡 Take-Home Messages:
- POD allows us to make statements about whether the learned eigenfunctions exhibit partial orthogonality, even in the absence of explicit orthogonality constraints.
- POD tells us whether mode collapse occurs without additional structure (❓ what 'additional structure' specicially refer to?❓).

---
## 🔮 Future possible implementations
- noisy or unnormalized densities
- partial observation windows
- unknown normalization constants
- orthogonality constraints between $\psi_n(\theta,x)$ 
- add energy ordering regularization
- add symplectic loss
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
