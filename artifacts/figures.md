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
> 📃 Loss trajectories reveal which physical and geometric structures are easiest or hardest to reconcile simultaneously.

### 🗝️ Key Take-Aways
#### Multi-objective competition governs training
- The optimization problem is inherently coupled:
    - wavefunctions,
    - eigenvalues,
    - and the shared potential $V_\theta(x)$
  
   are updated simultaneously under competing geometric and physical constraints.
- Consequently, convergence reflects a negotiated balance between:
    - data consistency,
    - PDE residual minimization,
    - smoothness regularization,
    - and weighted orthonormal structure.
  
#### Observable consistency emerges earlier than full physics consistency
- The data mismatch term decreases rapidly during early training.
- In contrast, the Schrödinger residual decreases more gradually and remains finite throughout optimization.
- ✨ This suggests that coarse observable structure is easier to identify than exact operator consistency in the low-fidelity inverse setting.

#### Smoothness regularization introduces optimization stiffness
- The smoothness loss initially dominates the optimization landscape.
- Because the smoothness term depends on second spatial derivatives of the learned potential, it is highly sensitive to local curvature fluctuations and discretization effects.
- The large transient spike early in training suggests a rapid reconfiguration of the coupled operator-eigenfunction geometry as the model simultaneously adjusts normalization, orthogonality structure, and potential curvature.

#### Stable convergence does not imply exact recovery
- The total loss stabilizes despite persistent nonzero physics residuals.
- This behavior is expected in the presence of:
    - noisy observations,
    - finite-difference discretization,
    - weighted normalization,
    - and competing regularization objectives.
- ✨ Consequently, convergence should be interpreted as approximate structural consistency rather than exact operator reconstruction.

<p align="center">
  <img src="../assets/images/loss_table_extended.png"
       alt="Loss table."
       height="150">
</p>

---

## Figure 2 - Learned Potential $V_\theta(x)$ vs. Ground Truth Potential $V(x)$ (Harmonic Oscillator)
![Potential Functions](demo_visuals/learned_potential.png)
> 📃 The learned potential reproduces localized confinement structure and spectral consistency within regions supported by the learned eigenfunctions. However, the sigmoidal / piecewise-flat shape of $V_\theta(x)$ deviating substantially from expected quadratic shape of  the ground-truth harmonic potential.

### 🔑 Key Take-Aways
#### 1. Local structure is more identifiable than global structure
- The learned operator captures qualitative confinement behavior near the spatial regions where the learned wavefunctions possess significant probability mass.
- However, the recovered potential deviates strongly from the analytic harmonic oscillator outside of these regions, indicating that spectral observations alone do not uniquely determine the global operator geometry
#### 2. The inverse Schrödinger problem remains fundamentally non-unique
- Multiple distinct potentials may reproduce similar:
    - eigenvalue spectra,
    - probability densities,
    - and low-order spatial statistics.
- Consequently, agreement between observables does not guarantee pointwise recovery of the true underlying potential.
- The learned $V_\theta(x)$ therefore represents:
    - one spectrally compatible solution among many others,
    - rather than the unique physical potential.
#### 3. Constraint strength depends on wavefunction support
- Regions where:
  $$ |\psi_n^\theta(x)|^2 \approx 0$$
  provide weak information to the inverse problem.
- In these regions:
    - the physics residual contributes little,
    - observational density supervision weakens,
    - and smoothness regularization dominates.
- This produces:
    - boundary flattening,
    - offest drift,
    - and reduced geometric fidelity away from occupied spatial regions.
#### 4. Smoothness regularization stabilizes the inverse problem but biases geometry
- The smoothness penalty suppresses high-frequency artifacts and prevents unstable curvature oscillations in $V_\theta(x)$.
- However, smoothness regularization also biases the recovered operator family toward lower-curvature solutions, which may differ from the true harmonic potential while still reproducing similar spectral observations.

### ✖️ Failure Modes
#### 1. Spectrally consistent but geometrically incorrect operators
- The learned potential preserves aspects of the spectral structure while failing to recover the correct global operator geometry.
- ✨ This demonstrates that spectral consistency alone is insufficient for full operator identifiability.
#### 2. Boundary under-constraint
- The strongest deviations occur near the domain edges where:
  - wavefunction amplitudes are negligible,
  - residual constraints weaken,
  - and regularization dominates optimization.
- This reflects a fundamental identifiability limitation of inverse spectral learning under finite spatial support.
---

> 🏡 Together, Figure 1 and 2 suggest the low-fidelity PINN converges toward a spectrally stable and internally consistent operator geometry, even when the recovered potential differs substantially from the ground-truth solution.
---

## Figure 3 - Learned Wavefunctions $\psi_n^\theta(x)$ vs. Ground Truth Wavefunctions $\psi_n(x)$ (Quantum Harmonic Oscillator)
![Learned Wavefunctions](demo_visuals/learned_wavefunctions.png)

> 🏡 The learned eigenfunctions (dashed green curves) preserve qualitative modal organization and nodal ordering while deviating substantially from the localized Hermite-Gaussian structure of the true harmonic oscillator eigenstates (pink curves).
> This is indicative of convergence toward a spectrally self-consistent but geometrically distorted operator family reminiscent of Fourier modes.

### 🔑 Key Take-Aways
- **Preserved spectral structure:** Learned modes maintain oscillatory complexity scaling, approximate parity, and nodal ordering despite inaccurate potential geometry.
- **Substantial geometric deviation:** Learned eigenfunctions $\psi_n^\theta(x)$ exhibit:
  - broader oscillations
  - reduced confinement
  - Fourier-like standing-wave behavior
- **Operator-eigenfunction coupling:** 
  - Deviation from the expected localized Hermite-Gaussian structure is consistent with the flattened learned potential $V_\theta(x)$, which fails to recover the quadratic structure of the harmonic oscillator potential.  
  - Such flattened potential curves produces globally oscillatory solutions rather than the expected localized bound states, maintaining internal consistency despite deviation from the ground truth system.
- **Regularization bias:** Smoothness priors favor lower-curvature geometries. This produces spatially smoother, less localized modes that are not consistent with the ground-truth wavefuncitons, yet they still satisfy Schrödinger residuals and observational constraints. This is a hallmark manifestation of the ill-posedness for the inverse Schrödinger problem.

### ✖️ Failure Modes
- Inaccurate operator recovery provides spectrally organized but physically incorrect eigenstates emerge
- Loss minimization succeeds yet fails to recover the ground truth potential geometry.
  - ✨ This highlights the fact that optimization stability does not guarantee unique physical recovery. This is the agrees with the fact that the inverse Schrödinger problem is ill-posed.
---

## Figure 4 - Learned vs. Ground Truth Energy Eigenvalues 
![Learned Energies](demo_visuals/learned_energies.png)
> 📃 The learned operator preserves spectral spacing and ordering across the first three wavefunctions. This indicates no mode swapping or spectral collapse occurred during training.

🎗️ Recall proper energy ordering is softly enforced by the ordering loss term $\mathcal{L}_\text{order}$.

--
> 🏠  **Spectral vs. geometric fidelity:** Together, Figures 2, 3, and 4 show that the model preserves eigenvalue ordering and modal hierarchy, but fails to recover confinement strength, Gaussian envelopes, and correct operator geometry. Importantly, this demonstrates that spectral agreement alone cannot uniquely reconstruct the ground truth phsyical operator.
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
