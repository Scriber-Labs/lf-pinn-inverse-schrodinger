# inverse-piml-schrodinger

## Planned Repo Structure

```
inverse-piml-schrodinger/
├── README.md
├── requirements.txt
├── pyproject.toml
├── src/
│   ├── model.py          # neural network ansatz for V(x)
│   ├── physics.py        # TISE residual, data mismatch, scale-aware smoothness
│   ├── inverse.py        # inverse-specific losses
│   ├── pod.py            # POD decomposition + reconstruction
│   ├── train.py          # training loop and CLI
│   └── utils.py          # helper functions (e.g., seeding and grids)
├── notebooks/
│   └── demo.ipynb        # visual + narrative
├── references/
│   ├── README.md         # a blank README.md idk what for though. chat gpt told me to include to when i decided to make the references folder
│   └── references.bib    # sources (❓🙋🏻‍♀️ not necessarily used explicity by the github repo code, but that's mainly because i have no idea how to do that. if there is a standard way of implementing bib files easily like in overleaf im open to something like that... just no relearning entire languages and new software related bullshit (pardon my French)
├── assets/
│   ├── images/
│   │   └── interpretability_axis.png    # ⚠️ check spelling
└── artifacts/
    ├── notes.md                         # conceptual notes and reflection
    ├── figures.md                       # structure-based analysis of demo visuals
    └── demo_visuals/                          
```

---
This project extends the `lf-pinn-harmonic-oscillator` framework to an inverse quantum problem.
While project 1 investigated the robustness  of physics-informed neural networks (PINNs) under low fidelity discretization for a *known Hamiltonian*, this project is concerned with the information about an *unknown potential* that can be recovered from partial, noisy observations of quantum states.

Using the time-independent Schrodinger equation (TISE) as a hard physics constraint, we treat the potential $V(x)$ as a learnable function while wavefunctions act as auxiliary fields constrained by the PDE. The model is trained using noisy spectral data and probability densities, mimicking low-fidelity experimental measurements.

As with project 1, the goal is not high-precision reconstruction, but interpretability and identifability. 
> Specifically, the aim of this repository is to understand which operator features are robustly recoverable under strong physics priors and limited data.

---

## Loss Function
📝 For this repo, we will assume:
- atomic units and normalized parameters (thus, $\hbar=1$ and $m=1$ electron rest mass)
- $\psi_n$ normalization is handled either implicitly or by the PDE
- probability density observations are on an absolute scale

🔮 Later, can include:
- noisy or unnormalized densities
- partial observation windows
- unknown normalization constants
- orthogonality constraints between $\psi_n(x)$ 
- add energy ordering regularization

### TISE Residual
$$\mathcal{L}_\text{TISE}=\Bigg<\bigg(-\frac{\hbar^2}{2m}\psi_n''(x)+V(x)\psi_n(x)-E_n\psi_n(x)\bigg)^2\Bigg>$$

### Scale-Aware Smoothness
$$\mathcal{L}_\text{smooth}=\Bigg<\frac{|V''(x)|^2}{\epsilon + |V(x)|^2}\Bigg>$$

### Data Mismatch (Noisy Observations)
$$\mathcal{L}_{data} = \sum _n {\Big|\Big| E_n - E_n^\text{obs} \Big|\Big|^2 + \Big|\Big| \Big( |\psi_n |^2 - \rho_n^\text{obs} \Big)}\Big|\Big|^2 $$

### Total Loss
$$\mathcal{L}_\text{total}=\lambda_\text{data}\mathcal{L}_\text{data}+\lambda_\text{phys}\mathcal{L}_\text{TISE}+\lambda_\text{smooth}\mathcal{L}_\text{smooth}$$

---
## Proper Orthogonal Decomposition (POD)
> POD does not enforce physics. It reveals structure.

### Specific Questions POD Answers

| Question                             | Why it matters                  |
| ------------------------------------ | ------------------------------- |
| Are ψₙ distinct or collapsing?       | Detects mode collapse           |
| How many effective modes exist?      | Identifiability                 |
| Are learned states redundant?        | Overparameterization            |
| Do modes align with energy ordering? | Model consistency               |
| Is orthogonality emerging naturally? | Strength of physics constraints |

### Important Notes
- POD allows us to make statements about whether the learned eigenfunctions exhibit partial orthogonality, even in the absence of explicit orthogonality constraints.
- POD tells us whether mode collapse occurs without additional structure (❓).

---
## ✅ To Do
- [ ] Figure out if/how you should implement grid-awareness (`make_grid()` function in `src/utils.py`) into `potential_smooth_loss()` from `src/physics.py`.
- [ ] Do I need to polish `train.py` regarding `make_grid()` stuff?
- [ ] Determine whether `model_derivative()` from `src/utils.py` has any practical use for this repo.
- [ ] `demo.ipynb` 
    - [ ] Setup
      - Load trained models
      - Define spatial grid
      - Set plotting utilities
    - [ ] Figure 1: Training curve
      - Plot
        - Total loss vs. epoch
        - Optional: physics vs. data vs. smoothness (stacked or separate)
      >🏡 *"The inverse problem stabilizes under competing physical and data-driven objectives."*
    - [ ] Figure 2: Learned potential
      - Plot
        - $V(x)$ vs. $x$
        - True $V(x)$ 
      - Message
        - Smoothness prior effect
        - Identifiability limits
        - Bias vs. variance tradeoff
      - Failure modes
        - Flattening
        - Over-smoothing
        - Boundary artifacts
    - [ ] Figure 3: Learned wavefunction $\psi_n(x)$
      - Plot
        - Learned (and ground truth ❓) $\psi_0(x)$, $\psi_1(x)$, and $\psi_2(x)$ (3 subplots)
      - Message
        - Phase ambiguity
        - Shape consistency
        - Node structure
      - ✨ Do not claim correctness; just structure
    - [ ] Figure 4: Probability densities $|\psi_n(x)|^2$ vs observations
      - Plot
        - $|\psi_n(x)|^2$
        - Observed $\rho_n(x)$
      - Message
        - Data anchoring
        - Indirect supervision
        - Why phase remains unconstrained
    - [ ] Figure 5: Overlap matrix heatmap (POD diagnostic)
      - Plot
        - Heatmap of $\langle \psi_m | \psi_n \rangle$
      - Message
        - Near-orthogonality emerges (or not)
        - Coupling through shared $V(x)$
      - ✨ This is very important for interpretability.
    - [ ] Figure 6: POD singular values
      - Plot
        - Singular values $\sigma_k$ vs. $k$ (log scale)
      - Message
        - Effective dimensionality
        - Redundant modes
        - Capacity vs. constraint
      - Failure mode
        - One dominant singular value -> collapse (❓)
    - [ ] Figure 7: POD modes vs. learned $\psi_n$ (optional if energy ordering is messy)
      - Plot
        - Compare POD spatial modes to $\psi_n$
      - Message
        - Learned basis is not equivalent to ground truth physical eigenbasis
        - Still explains data
- [ ] `figures.md`
  - Each figure should have:
    - Title
    - What is shown
    - What it tells us
    - Possible failure modes
    
    🔵 Example tone:
      > *"Figure 5 reveals partial orthogonality between learned eigenfunctions, despite no explicit orthogonality constraint. This suggests that the TISE residual alone imposes meaningful structure, though small off-diagonal overlaps indicate residual mode mixing."*
---
