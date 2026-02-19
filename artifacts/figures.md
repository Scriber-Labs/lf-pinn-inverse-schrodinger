# Figure Analysis
## Figure 1 - Training Curves
![Training Curves](demo_visuals/training_curves.png)

> 🏡 *"The inverse problem stabilizes under competing physical and data-driven objectives."*
![loss_table](./assets/images/loss_table_large.png)
---

## Figure 2 - Learned Potential $V_\theta(x)$ vs. Ground Truth Potential $V(x)$ (Harmonic Oscillator)
![Potential Functions](demo_visuals/learned_potential.png)

### 🏡 Take-Home Messages
- **Smoothness prior effect** pulls $V_\theta(x)$ towards a low-curvature shape.
- **Non-uniqueness of $V_\theta$**: The same set of eigenstates (wavefunctions and their associated eigenvalue energies) can be produced by more than one $V(x)$ $\rightarrow$ Thus, the inverse Schrodinger problem is fundamentally ill-posed.
  - $\mathcal{L}_\text{smooth}$ helps guide the model to a physically plausible solution (e.g., no sharpe curves).
  - ⚠️ However, $\mathcal{L}_\text{smooth}$ does not guarantee uniqueness!!
### ✖️ Failure Modes
- **Bias vs. variance tradeoff:** 
  - If $\lambda_\text{smooth}$ is too strong, flattening (bias) occurs.
    - **Over-smoothing** (bias) occurs if $\lambda_\text{smooth}$ is too strong.
    - In the extreme case, **flattening** (bias) occurs and $V_\theta \rightarrow \text{const}$.
  - If $\lambda_\text{smooth}$ is too weak, noisy perturbations (wiggles) manifest.
  - 
- **Boundary artifacts** are more likely to manifest due to the model being less constricted near the boundaries.

---

## Figure 3 - Learned Wavefunctions $\psi_n^\theta(x)$ vs. Ground Truth Wavefunctions $\psi_n(x)$ (Quantum Harmonic Oscillator)
![Learned Wavefunctions](demo_visuals/learned_wavefunctions.png)

### 🏡 Take-Home Messages
- **Phase ambiguity:** overall sign may flip 
- **Shape consistency:** learned curves retain the same envelope
- **Node structure:** zeros line up with the true wavefunctions.

---

## Figure 4 - Learned vs. Ground Truth Energy Eigenvalues 
![Learned Energies](demo_visuals/learned_energies.png)
> A bar chart of first three learned energy eigenvalues $E_n^\theta$ vs. ground truth energy eigenvalues $E_n$ for the quantum harmonic oscillator.

---

## Figure 5 - Learned vs. (Dummy) Observed Probability Densities
![Probability Densities](demo_visuals/density.png)
> Inferred probability densities $|\psi_n^\theta(x)|^2$ vs. fake observed probability densities $\rho_n^\text{obs}(x)$ for the first three eigenmodes.

### 🏡 Take-Home Messages
- **Data anchoring**
- **Indirect supervision**
- **Why phase remains unconstrained**


---
# POD Diagnostics

> 💡**Big Idea:** POD does not enforce physics. It reveals structure.


### Specific Questions POD Answers

| 🧙🏻‍♂️ Question                                                                            | ✨ Relvance                     |
|---------------------------------------------------------------------------------------------| ------------------------------- |
| Are $\psi_n^\theta(x)$ distinct or collapsing?                                              | Detects mode collapse           |
| How many effective modes exist?                                                             | Identifiability                 |
| Are learned states (❓is this just the same thing as saying 'learned eigenmodes') redundant? | Overparameterization            |
| Do modes align with energy ordering?                                                        | Model consistency               |
| Is orthogonality emerging naturally?                                                        | Strength of physics constraints |

### 🏡 Take-Home Messages:
- POD allows us to make statements about whether the learned eigenfunctions exhibit partial orthogonality, even in the absence of explicit orthogonality constraints.
- POD tells us whether mode collapse occurs without additional structure (❓ what 'additional structure' specicially refer to?❓).


## Figure 6 - Overlap Heatmap
![Overlap Heatmap](demo_visuals/overlap_heatmap.png)

### 🏡 Take-Home Messages
- **Near-orthogonality emerges** $\rightarrow$ agrees with ground truth wavefunctions.
- **Coupling** through shared $V_\theta(x)$.
  - Specifically, All $\psi_n^\theta$ are learned simultaneously via a *single shared* potential $V_\theta(x)$
  - Thus, any error in $V_\theta(x)$ will 'leak' into $\psi_n^\theta$. "❓❓❓❓❓Any error in $V_\theta$ couples all states together; if the potential is wrong in a region, it distorts all modes that have support there. This is why off-diagonal overlaps are possible even without an explicit interaction term ❓❓❓❓❓"

✨ This is important for interpretability!

---

## Figure 7 - POD Diagnostics 
### Figure 7a - POD Singular Values
![POD Singular Values](demo_visuals/pod_singular_values.png)
> Singular values $\sigma_k$ vs. $k$ (logarithmic scale).

### Figure 7b - POD Spatial Modes vs. Learned Wavefunctions vs. Ground Truth Wavefunctions
![POD Spatial Modes](demo_visuals/pod_modes.png)
> Compare first three POD spatial modes to $\psi_n^\theta(x)$ and $\psi_n(x)$.

### 🏡 Take-Home Messages
- Learned basis is not equivalent to ground-truth physical eigenbasis
- Still explains the data

## Figure 7c - $\langle u_k | \psi_n^\theta\rangle$ Overlap Matrix
![Cross Overlap Heat Matrix](demo_visuals/cross_overlap_heatmap.png)
Columns correspond to learned wavefunctions $\psi_n^\theta$ and rows correspond to POD eigenmodes $u_k$.