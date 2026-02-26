# Figure Analysis
## Figure 1 - Training Curves
![Training Curves](demo_visuals/training_curves.png)

> 🏡 The inverse problem converges to a stable multi-objective equilibrium. Physics residual and data mismatch are minimized concurrently, while normalization and smoothness constraint terms regulate solution geometry.
> 
> Training dynamics exhibit staged constraint resolution:
>   - Early objective competition
>   - Mid-stage geometry stabilization
>   - Late-stage oscillatory constraint enforcement as normalization corrections stabilize the solution manifold.
>     - 📝 Similar staged dynamics were observed in [Project 1 (Figure 1)](https://github.com/Scriber-Labs/lf-pinn-harmonic-oscillator/blob/main/artifacts/figures.md), suggesting consistent constraint geometry behavior across problem classes.

<p align="center">
  <img src="../assets/images/loss_table_large.png"
       alt="Loss table."
       height="300">
</p>

---

## Figure 2 - Learned Potential $V_\theta(x)$ vs. Ground Truth Potential $V(x)$ (Harmonic Oscillator)
![Potential Functions](demo_visuals/learned_potential.png)

### 🔑 Key Take-Aways
- **Smoothness prior effect:**  Regularization of $\mathcal{L}_\text{smooth}$ encourages low-curvature solutions, guiding $V_\theta(x)$ toward a geometrically stable solution state.
- **Inverse problem non-uniqueness:** The inverse Schrödinger problem is fundamentally ill-posed. Specifically, the same finite set of eigenfunctions and associated eigenvalues can be produced by multiple potentials.
  - $\mathcal{L}_\text{smooth}$ helps guide the model to a physically plausible solution (e.g., no sharp curvature artifacts).
  - However, $\mathcal{L}_\text{smooth}$ does not guarantee uniqueness!!
- **Domain-dependent identifiability:** Divergence near boundaries arise due to the wavefunctions having negligible amplitude in those regions. 
  - This results in:
    - Numerical weakening of the physics residual.
    - Data set provides minimal constraint in these regions.
    - The smoothness term biases the solution toward flattening.
  > 🏡 Thus, the potential is identifiable primarily within the spectral support of the trained eigenstates.

### ✖️ Failure Modes
- **Bias vs. variance tradeoff:** 
  - If $\lambda_\text{smooth}$ is too large, **over-smoothing** (bias) occurs. In the extreme case, **flattening** occurs and $V_\theta \rightarrow \text{const}$.
  - If $\lambda_\text{smooth}$ is too small, noisy perturbations and high-frequency artifacts emerge in $V_\theta(x)$.
- **Boundary artifacts** are more likely to manifest due to the model being less constricted near the boundaries (see domain-dependent identifiability).


---

> 🏡 Together, Figure 1 and Figure 2 suggest the low-fidelity PINN formulation aligns operator spectrum, solution support, and constraint geometry into a stable, interpretable equilibrium.
---

## Figure 3 - Learned Wavefunctions $\psi_n^\theta(x)$ vs. Ground Truth Wavefunctions $\psi_n(x)$ (Quantum Harmonic Oscillator)
![Learned Wavefunctions](demo_visuals/learned_wavefunctions.png)

> 🏡 The learned operator demonstrates spectral alignment across multiple eigenmodes, recovering correct parity, node structure, and envelope geometry.


### 🔑 Key Take-Aways
- **Phase ambiguity:** Overall sign flips are physically irrelevant due to global phase invariance.
- **Shape consistency:** Learned eigenfunctions retain correct Gaussian envelope structure.
- **Node structure:** Zeros align accurately with ground truth analytical solutions. This indicates correct operator curvature.

---

## Figure 4 - Learned vs. Ground Truth Energy Eigenvalues 
![Learned Energies](demo_visuals/learned_energies.png)
> 🏡 The learned operator preserves spectral spacing and ordering across the first three wavefunctions.

### 🔑 Key Take-Aways
- **Energy ordering preserved:** No mode swapping or spectral collapse.
- **Approximate linear spacing retained:** Indicates correct quadratic curvature in $V_\theta(x)$.
- **Small systematic bias:** Slight underestimation of extreme modes suggests mild curvature underfitting. This is consistent with smoothness regularization.
---

## Figure 5 - Learned vs. Noisy Observed Probability Densities
![Probability Densities](demo_visuals/density.png)
> 🏡 Inferred probability densities $|\psi_n^\theta(x)|^2$ vs. fake observed probability densities $\rho_n^\text{obs}(x)$ for the first three eigenmodes. Despite observational noise, the learned operator preserves node structure, parity, and multi-lobe geometry across modes $n=0,1,2$.

### 🔑 Key Take-Aways:
- **Node alignment:** Zero crossings are accurately recovered despite noisy (indirect) supervision.
- **Spectral geometry preservation:** Lobe count and parity structure match analytical solutions.
- **Controlled amplitude bias:** Slight peak underestimation and mild broadening are consistent with regularization under noisy data.
- **Observable robustness:** Agreement in $|\psi|^2$ indicates stable operator recovery from corrupted measurements.
- **Data anchoring:** The observational density loss term prevents arbitrary drift in function space by constraining the learned states to match measurable structure. Note this is a *partial* anchor; not a full identification constraint.
- **Indirect supervision:** The model must $\psi$ such that its squared magnitude matches data, while also satisfying the PDE constraint. 
- **Phase remains unconstrained:** Loss is invariant under $\psi \rightarrow -\psi$.


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