# inverse-piml-schrodinger

## Repo Structure

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

## 🌍 Global Design Choices
Assumptions:
- Atomic units: $\hbar = m = 1$

What is being learned:
- $V_\theta(x)$ via the MLP.
- ${\psi_\theta}_n(x)$ via the MLP. (‼️Get feedback about the notation we use to write the learned wave function; usually we just assume $\psi_n(x) \equiv {\psi_\theta}_n(x)$, but the ambiguity bothers me; I just don't know what to do about the indices or whether there are already best practice  guidelines for these sorts of things).
- ${E_\theta}_n$ as learnable scalars. (‼️ same feedback is needed as requested for the learned wave function).

Orthogonality
- For our low fidelity design, we are not _enforcing_ orthogonality directly.
- However, our POD function (`src/pod.py`) allows us to _diagnose_ orthogonality.

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
$$\mathcal{L}_\text{data} = \sum _n {\Big|\Big| E_n - E_n^\text{obs} \Big|\Big|^2 + \Big|\Big| \Big( |{\psi_\theta}_n |^2 - \rho_n^\text{obs} \Big)}\Big|\Big|^2 $$

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
- [ ] `demo.ipynb`
- [ ] `figures.md`
  - Each figure should have:
    - Title
    - What is shown
    - What it tells us
    - Possible failure modes
    
    🔵 Example tone:
      > *"Figure 5 reveals partial orthogonality between learned eigenfunctions, despite no explicit orthogonality constraint. This suggests that the TISE residual alone imposes meaningful structure, though small off-diagonal overlaps indicate residual mode mixing."*
---
