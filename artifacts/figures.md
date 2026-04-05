# Figure Analysis
## 🧙‍♂️ Questions That Need Answering
### 🔷 Identifiability
- [ ] Which features of $V(x)$ are uniquely recoverable?
- [ ] Does smoothness regularization bias the recovered potential family?
### 🔷 Mode Structure
- [ ] Does orthogonality emerge without reinforcement?
- [ ] Does POD reveal effective low-rank eigenspaces?
### 🔷 Inverse Stability
- [ ] Does noise induce mode mixing?
- [ ] Are certain eigenstates more stable under low-fidelity observation?
### 🔷 Structural Recovery
- [ ] Is curvature recoverable before amplitude?
- [ ] Are nodal locations more identifiable than potential amplitude?


---
## Figure 1 - Training Curves
![Training Curves](demo_visuals/training_curves.png)
> 🏡 ENTER FIGURE 1 TAKE-HOME MESSAGE HERE

<p align="center">
  <img src="../assets/images/loss_table_extended.png"
       alt="Loss table."
       height="150">
</p>

---

## Figure 2 - Learned Potential $V_\theta(x)$ vs. Ground Truth Potential $V(x)$ (Harmonic Oscillator)
![Potential Functions](demo_visuals/learned_potential.png)
> 🏡 ENTER FIGURE 2 TAKE-HOME MESSAGE HERE

### 🔑 Key Take-Aways
- **Smoothness prior effect:**  Regularization of $\mathcal{L}_\text{smooth}$ encourages low-curvature solutions, guiding $V_\theta(x)$ toward a geometrically stable solution state.
- **Inverse problem non-uniqueness:** The inverse Schrödinger problem is fundamentally ill-posed. Specifically, the same finite set of eigenfunctions and associated eigenvalues can be produced by multiple potentials.
  - $\mathcal{L}_\text{smooth}$ helps guide the model to a physically plausible solution (e.g., no sharp curvature artifacts).
  - However, $\mathcal{L}_\text{smooth}$ does not guarantee uniqueness!
- **Domain-dependent identifiability:** Divergence near boundaries arise due to the wavefunctions having negligible amplitude in those regions. 
  - This results in:
    - Numerical weakening of the physics residual.
    - Data set provides minimal constraint in these regions.
    - The smoothness term biases the solution toward flattening.
### ✖️ Failure Modes
- **Bias vs. variance tradeoff:** 
  - If $\lambda_\text{smooth}$ is too large, **over-smoothing** (bias) occurs. In the extreme case, **flattening** occurs and $V_\theta \rightarrow \text{const}$.
  - If $\lambda_\text{smooth}$ is too small, noisy perturbations and high-frequency artifacts emerge in $V_\theta(x)$.
- **Boundary artifacts** are more likely to manifest due to the model being less constricted near the boundaries (see domain-dependent identifiability).


---

> 🏡 ❓Together, Figure 1 and Figure 2 suggest the low-fidelity PINN formulation aligns operator spectrum, solution support, and constraint geometry into a stable, interpretable equilibrium.❓ DOUBLE CHECK
---

## Figure 3 - Learned Wavefunctions $\psi_n^\theta(x)$ vs. Ground Truth Wavefunctions $\psi_n(x)$ (Quantum Harmonic Oscillator)
![Learned Wavefunctions](demo_visuals/learned_wavefunctions.png)

> 🏡 ENTER FIGURE 3 TAKE-HOME MESSAGE HERE

### 🔑 Key Take-Aways
- **Phase ambiguity:** Overall sign flips are physically irrelevant due to global phase invariance.
- **Shape consistency:** Learned eigenfunctions retain correct Gaussian envelope structure.
- **Node structure:** Zeros align accurately with ground truth analytical solutions. This indicates correct operator curvature.

---

## Figure 4 - Learned vs. Ground Truth Energy Eigenvalues 
![Learned Energies](demo_visuals/learned_energies.png)
> 🏡 The learned operator preserves spectral spacing and ordering across the first three wavefunctions. This indicates no mode swapping or spectral collapse occurred during training.

🎗️ Recall proper energy ordering is softly enforced by the ordering loss term $\mathcal{L}_\text{order}$.

---

## Figure 5 - Learned vs. Noisy Observed Probability Densities
![Probability Densities](demo_visuals/density.png)
> 🏡 ENTER TAKE-HOME MESSAGE HERE

### 🔑 Key Take-Aways:
- **Data anchoring:** The observational density loss term prevents arbitrary drift in function space by constraining the learned states to match measurable structure. Note this is a *partial* anchor; not a full identification constraint.
- **Indirect supervision:** The model must $\psi$ such that its squared magnitude matches data, while also satisfying the PDE constraint. 
- **Phase remains unconstrained:** Loss is invariant under $\psi \rightarrow -\psi$.

---
# POD Diagnostics

> 💡**Big Idea:** POD does not enforce physics. It reveals structure.

---

## Figure 6a - POD Singular Values
![POD Singular Values](demo_visuals/pod_singular_values.png)
> 🏡 The POD singular values of the learned eigenmode matrix $\mathbf{\Psi^\theta}$ have values close to unity. This indicates that the learned eigenfunctions form a well-conditioned and nearly orthonormal basis. These results independently confirm the spectral consistency observed in the overlap matrix (Figure 6).

## Figure 6b - POD Spatial Modes vs. Learned Wavefunctions vs. Ground Truth Wavefunctions
![POD Spatial Modes](demo_visuals/pod_modes.png)
> 🏡 The POD spatial modes extracted from the learned eigenfunction matrix reveal that the dominant spatial patterns closely align with linear combinations of the learned eigenfunctions.

### 🔑 Key Take-Aways
#### Learned eigenfunctions span the dominant spatial subspace.
- The POD spatial modes represent the principal spatial patterns shared across the learned dataset $\mathbf{\Psi^\theta}$.
- These spatial modes appear manifest as linear combinations of the learned eigenfunctions.
- This indicates that the learned eigenbasis already spans the dominant spatial structures present in the system.
- 
#### Well-conditioned learned eigenbasis.
- The singular values are all similar in magnitude.
- No single spatial mode dominates the SVD representation.
- This indicates that the learned eigenfunctions form a well-conditioned basis with minimal redundancy.

#### Learned basis differs from THE physical eigenbasis but still captures the system.
- The POD spatial modes do not align with individual eigenfunctions.
- Instead, they emerge as mixtures that optimally represent the spatial variance in the learned dataset.
- Despite this rotation of the learned eigenbasis, the learned eigenfunctions still accurately reproduce the observed probability densities (Figure 5).

#### Consistency with the learned quantum operator
- Similarity between POD spatial modes and learned eigenfunctions suggests that the learned eigenbasis sufficiently captures the dominant spatial structures generated by $\hat{H_\theta}$.
- Together with the conclusion from Figure 7a (well-conditioned spectrum), this suggests that the learned operator is geometrically consistent.



## Overlap Heatmaps
### Figure 6c
<p align="center">
  <img src="demo_visuals/overlap_heatmap.png"
       alt="Loss table."
       height="500">
</p>

> 🏡 $\langle\psi_m^\theta | \psi_n^\theta\rangle \approx \delta_{mn}$ provides evidence that the learned operator remains spectrally consistent across modes. This indicates the learned model is not just interpolating; it is learning a coherent quantum operator. 

### 🔑 Key Take-Aways
#### Near-orthogonality emerges naturally 
  - The overlap heatmap tell us $\langle\psi_m^\theta | \psi_n^\theta\rangle \approx \delta_{mn}$.
  - These results agree with the expected structure of the eigenfunctions of a Hermitian Hamiltonian.
  - Explicit orthogonality constraints are absent from the PIML architecture.

#### Linear independence and distinctness of learned eigenfunctions
  - Nonzero determinant indicates linear independence.
  - Small off-diagonal values indicate distinct learned eigenfunctions.

#### Coupling through the shared potential $V_\theta(x)$

- All $\psi_n^\theta(x)$ are solutions of the same operator $$\hat{H}_\theta=-\frac{1}{2}\frac{\partial^2}{\partial x^2} + V_\theta(x) \, .$$ 
- Note that $\hat{H}_\theta$ is global across all modes.

#### Error propagation across eigenstates
  - Since the potential is shared, any error in $V_\theta(x_0)$ propagates to every eigenfunction with support near $x_0$.
  - This induces correlated errors across learned eigenstates, which may manifest as
    - Slight systematic energy bias.
    - Consistent wavefunction broadening across learned eigenfunctions.
    - Non-zero off-diagonal overlaps (if $\hat{H}_\theta$ is poorly learned).



## Figure 6C - $\langle u_k | \psi_n^\theta\rangle$ Overlap Matrix
<p align="center">
  <img src="demo_visuals/cross_overlap_heatmap.png"
       alt="Loss table."
       height="500">
</p>

> 📝 Columns correspond to learned wavefunctions $\psi_n^\theta$ and rows correspond to POD eigenmodes $u_k$.

### 🔑 Key Take-Aways

#### POD mode 0

$$u_0 \approx 0.97\psi_2^\theta$$
- Tells us the learned eigenstate with the highest energy contributes the largest spatial variance in the dataset.
- This agrees with physical ground truth since higher eigenstates oscillate more.

#### POD mode 1

$$u_1 \approx -0.79\psi_1^\theta - 0.57\psi_2^\theta$$
- Tells us this mode is primarily a mixture of excited learned eigenstates.

#### POD mode 2

$$u_2\approx -0.81\psi_0^\theta$$
- Tells us the learned ground-state contributes the least spatial variance in the dataset.
- This agrees with the physical ground truth since lower eigenstates oscillate less.

# Conclusions (POD Analysis)
| **Diagnostic** | **Measurement** |
|----------------|-----------------|
| Figure 6 | orthogonality of learned eigenfunctions |
| Figure 7a | conditioning of eigenbasis |
| Figure 7b | spatial structure of modes |
| Figure 7c | relationship between POD modes and learned eigenbasis|

### Specific Questions POD Answers

| 🧙🏻‍♂️ Question                                                                                                      | ✨ Relevance            | ✔️ Answer     | 🖼️ Figure(s) | 💬 Comments                                                                                                                                                                                                                                                                                        | 🧠 Interpratibility                                                             |
|-----------------------------------------------------------------------------------------------------------------------|------------------------|---------------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| Are $\psi_n^\theta(x)$ distinct or collapsing?                                                                        | Detects mode collapse  | distinct      | Figure 6      | Flattening of learned wavefunctions occurs without wavefunction normalization loss term. <br/> <br/> Moreover, the smoothness regularization term keeps the learned potential structurally constrained so that curvature isn't too steep.                                                          | Orthogonality is (approximately) observed in learned wavefunctions.             |
| How many effective modes exist?                                                                                       | Identifiability        | 3             | Figure 6      | All singluar values are near unity.                                                                                                                                                                                                                                                                | No single spatial mode dominates the data matrix.                               |
| Are learned eigenstates redundant?                                                                                    | Overparameterization   | not redundant |               |                                                                                                                                                                                                                                                                                                    |                                                                                 |
| Do spatial modes align with energy ordering? | Model consistency      |               |               |                                                                                                                                                                                                                                                                                                    |                                                                                 |
| Is orthogonality emerging naturally?                                                                                  | Structural consistency | yes           | Figure 6      | Importantly, this architecture does NOT enforce $\langle \psi_m^\theta , \psi_n^\theta \rangle=\delta_{mn}$. However, we observe near-orthogonality in Figure 6. <br/> <br/>This result is structurally consistent with Hermitian operators (orthogonal eigenfunctions with distinct eigenvalues). | The learned potential is consistent enough to preserve orthogonality structure. |

---
## ✨ General Insights (Figures 2-7)
> #### 🧠 Big Interpretation Insight: Three (Four ❓) 'layers' of structure

| **Layer**          | **Interpretation**     | **📝 Notes**                                           |
|--------------------|------------------------|--------------------------------------------------------|
| 1️⃣                | data fidelity          |                                                        |
| 2️⃣                | observable consistency | $\|\psi_n^\theta \|^2$ matches noisy data              |
| 3️⃣                | spectral geometry      | nodes, parity, and ordering are preserved              |
| 4️⃣  POD analysis (❓) | operator coherence     | near-orthogonality and energy spacing naturally emerge |
