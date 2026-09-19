# Figure Analysis (macOS Computer)

## Figure 1: Training Curves
![Training-curve panel](demo_visuals/training_curves.png)
![Training-curve zoomed](demo_visuals/training_curves_zoomed.png)
![Training-curve zoomed spike 1](demo_visuals/training_curves_spike_1_epoch_5.png)
![Training-curve zoomed spike 2](demo_visuals/training_curves_spike_2_epoch_782.png)

> 🏡 The optimizer exhibits three distinct regime transitions before settling on a stable plateau after around epoch 800.

> 🔑 **Key Insights**
> 1. **Epoch 5:** Expected transient while the network adjusts from random initial weights.
> 2. **Epoch 782:** Discovery of a higher-curvature potential; smoothness and total loss spike while physics and data terms rise only moderately.

> 2. ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description**                                                                                                                         | **Explanation**                                                                                                                                                                                                                                                          |
> | :---------- | :--------------- |:----------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
> | ❌ | High final loss | Optimizer stalls in a local minimum. <br> <br> Heavily driven by the penalty weighting $\lambda_\text{smooth} \gg \lambda_\text{data}$. | Total loss remains greater than 1e-1 at epoch 6000. <br> <br> Note that a high wieghted total loss is not necessarily a failure if the physical residue $\mathcal{L}_\text{SE}$ and data loss $\mathcal{L}_\text{data}$ are near convergence ($~10^{-3}$ to $~10^{-4}$). |
> | ✔️ | Oscillation avoided | Unbalanced loss weights can cause loss terms to oscillate.                                                                              | Curves converge monotonically shortly after epoch 782.                                                                                                                                                                                                                             |
> | ❌ | Physics collapse | Data loss decreases, while TISE residual increases.                                                                                     | Indicates operator inconsistency.                                                                                                                                                                                                                                        |
> | ❌ | Over-regularization | Smoothness term dominates, spectrum becomes inacurate.                                                                                  | Loss curves all begin to plateau after ~ epoch 850 with $\lambda_\text{smooth}$ taking on values much greater than others.                                                                                                                                                                                          | 

## Figure 2: Learned vs. true potential
![Learned potentials](demo_visuals/learned_potential.png)

> 🏡 The learned potential $V_\theta(x)$ (sigmoidal) differs markedly from the harmonic ground truth $V(x)=\tfrac12 x^2$.

> 🔑 **Key Insights**
> 1. Central regions of the learned eigenfunctions (Fig. 3) and densities (Fig. 5) match the ground truth far better than the tails.
> 2. The model learns only the portion of $H_\theta$ required to reproduce high-probability regions, exposing the inverse problem's under-determinism.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode**          | **Description**                                                                          | **Explanation**                                                    |
> |:------------|:--------------------------|:-----------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
> | ❌           | Geometric mismatch        | Learned $V_\theta$ shape incompatible with true quadratic.                               | Central well too narrow; tails saturate at $V_\theta \approx \pm 12 $. |
> | ❌           | Boundary under-constraint | Sparse data at $x \in (-\infty, -4.5] \cup [4.5, \infty)$ allows the potential to drift. | Grey dashed domain limits show no training points beyond. |


## Figure 3: Learned wavefunctions
![Learned wavefunctions](demo_visuals/learned_wavefunctions.png)

> 🏡 Learned eigenfunctions $\psi_n^\theta(x)$ capture the nodal pattern but diverge in low-amplitude tail regions.

> 🔑 **Key Insights**
> 1. **Phase matching** - Correct nodal count confirms energy ordering.
> 2. **Central accuracy** - Highest fidelity occurs where $|\psi_n|^2$ is largest.
> 3. **Tail divergence** - For $x \in (-4.5, -2] \cup [2, 4.5)$, the learned curves overshoot, reflecting data scarcity.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation**                            |
> | :---------- | :--------------- | :-------------- |:-------------------------------------------|
> | ❌ | Nodal mis-count | Extra nodes appear beyond $x \approx \pm 3$) | Indicates spectral leakage.                |
> | ✔️ | Sign / parity flip | Unaligned solutions may invert parity | Sign aligned; parity matches ground truth. |
> | ❌ | Spurious oscillations | High-frequency ripples in tails from weak $V_\theta$ smoothness. | Visible beyond $x\approx \pm 4$.           |

## Figure 4: Energy eigenvalues
![Energy spectrum comparison](demo_visuals/learned_energies.png)

> 🏡 Learned energies $E_n^\theta$ follow the harmonic spectrum $E_n=n+\tfrac12$ and match observations within 5 %.

> 🔑 **Key Insights**
> 1. Correct ordering suggests $\mathcal{L}_\text{order}$ is effective.
> 2. Spectrum remains stable despite 2 % Gaussian noise in training data.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | Spectral fit, wrong operator | Energies match, but $V_\theta$ deviates (see Fig. 2) |

## Figure 5: Learned density vs. observations
![Probability-density comparison](demo_visuals/density.png)

> 🏡 Learned densities, $\rho_n^\theta = |\psi_n^\theta|^2$ agree with 2 %-noise observations.

> 🔑 **Key Insights**
> 1. **Noise filtering** - PINN acts as a physics-informed smoother.
> 2. **Data dominance** - Good density fit persists even with incorrect potential (Fig. 2), confirming $\mathcal{L}_\text{data}$ is easy to minimize.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ✔️ | Peak flattening | Excessive $\lambda_\text{smooth}$ can lower peaks | Peaks are preserved $\Rightarrow$ smoothing is well-tuned. |
> | ✔️ | Mode merging | Energy mis-ordering can collapse multiple states onto one density. |

## Figure 6: POD singular values 
![POD singular values](demo_visuals/pod_singular_values.png)

> 🏡 Singular values from the POD of the learned wavefunction matrix decrease (log scale) from $\approx 1$.

> 🔑 **Key Insights**
> 1. **Rank efficiency** - Rapid two-decade decay indicates a low-dimensional basis.
> 2. **Basis conditioning** - Separation between $\sigma_0$, $\sigma_1$, and $\sigma_2$ quantifies how much "physics" each node carries.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation**                                            |
> | :---------- | :--------------- | :-------------- |:-----------------------------------------------------------|
> | ❌ | Flat spectrum | All $\sigma_i$ nearly equal; no dominant low-rank modes. | **Noise-dominated snapshopts or over parameterization.** The PINN is outputting random high-frequency noise or unconstrained oscillations rather than smooth quantum states. |
> | ❌ | Slow decay | $\tfrac{\sigma_{2}}{\sigma_{0}} \geq 0.3 \Rightarrow$ redundant or correlated modes. | **Under-fitting or unresolved high-frequency physics.** The PINN is struggling to resolve sharp potential barriers or fine features, scattering energy across many modes rather than capturing it in the primary states. |

## Figure 7: Spatial overlap heatmap
![Wavefunction overlap heatmap](demo_visuals/overlap_heatmap.png)

> 🏡 Mutual inner product matrix $\langle \hat{\psi}_m^\theta | \hat{\psi}_n^\theta \rangle$ forms an exact identity matrix.

> 🔑 **Key Insights**
> 1. **Strict Mutual Orthogonality:** - Diagonals entries are identically $1.00$ and all off-diagonal entries are $0.00$.
> 2. **Hermitian Basis Property:** - Learned eigenfunctions constitute a numerically orthonormal spatial set.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ✔️ | Non-orthogonality | Off-diagonal entries exceed $0.10$. | Here, max off-diagonal is $0.00$ confirming orthonormal learned state representation. | 


## Figure 8: POD spatial modes
![POD spatial modes](demo_visuals/pod_modes.png)

> 🏡 POD spatial modes $u_k(x)$ deviate from physical eigenfunctions due to spatial mode mixing.

> 🔑 **Key Insights**
> 1. **Spatial Shift:** POD mode $u_0(x)$ is shifted horizontally relative to symmetric ground truth $\psi_0(x)$.
> 2. **Asymmetric Amplitude:** POD mode $u_1(x)$ exhibits asymmetric peak/trough amplitudes ($-0.8$ vs. $+0.45$).
> 3. **Mixed Coordinate Frame:** SVD modes represent linear combinations of learned states rather than pure eigenstates.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | Mode mixing | Pod spatial modes fail to align with pure physical eigenfunctions. | Significant spatial distortion and asymmetry in $u_0, u_1$.|


## Figure 9: Cross-overlap heatmap (POD vs. learned)
![Cross overlap: POD vs. learned](demo_visuals/cross_overlap_heatmap.png)

> 🏡 Cross-projections $\langle u_k | \hat{\psi}_0^\theta \rangle$ exhibit strong non-diagonal coupling between POD modes
> and learned wavefunctions.

> 🔑 **Key Insights**
> 1. **Rotated Basis:** Primary projections ($\langle u_0 | \hat{\psi}_0^\theta \rangle=0.88$, $\langle u_1 | \hat{\psi}_1^\theta \rangle=0.87$, $\langle u_2 | \hat{\psi}_2^\theta \rangle=0.98$).
> 2. **Off-Diagonal Cross-Talk:** Significant off-diagonal components ($\langle u_0 | \hat{\psi}_1^\theta \rangle =0.46$, $\langle u_1 | \hat{\psi}_0^\theta \rangle=-0.47$, $\langle u_2 | \hat{\psi}_1^\theta \rangle=-0.20$).

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description**                                           | **Explanation** |
> | :---------- | :--------------- |:----------------------------------------------------------| :-------------- |
> | ❌ | Distributed overlap | Non-diagonal matrix entries exceed tolerance. | Caused by missing $\sqrt{w\Delta x} (❓)$ <br> <br> Off-diagonals reach magnitudes up to $0.47$, confirming basis rotation.|

## Figure 10: POD Eigen-Alignment
![POD Eigen-Alignment](demo_visuals/pod_eigen_alignment.png)

> 🏡 Direct overlap $\langle u_k | \psi_n \rangle$ between POD modes and ground truth eigenfunctions indicate imperfect physical recovery.

> 🔑 **Key Insights**
> 1. **Diagonal Attenuation:** Overlap values along the diagonal are $0.82 \, (k=0, n=0)$, $0.69 \, (k=1, n=1)$, and $0.70 \, (k=2, n=2)$.
> 2. **Physical Cross-Talk:** Substantial projection onto adjacent physical eigenstates ($-0.44$ and $+0.38$).

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | Mis-alignment | Off-diagonal $> 0.2$ or diagonals $<0.90$ indicates POD not yet physical. | Off-diagonals reach $-0.44$ and diagonals drop to $0.69$. |


## Figure 11: POD Temporal Modes
![POD temporal modes](demo_visuals/pod_temporal_modes.png)

> 🏡 Right singular matrix components $V_{nk}$ reflect modal participation of POD basis vectors across learned states.

> 🔑 **Key Insights**
> 1. **Modal Composition:** State $0$ draws from $u_0$ ($-0.88$) and $u_1$ ($-0.47$); State $1$ draws from $u_0$ ($-0.46$) and $u_1$ ($+0.87$).
> 2. **State Decoupling:** State $2$ is predominantly aligned with $u_2$ ($+0.98$).

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | Incoherent coefficients | Scatter of non-zero coefficients across temporal mode entries. | States $0$ and $1$ exhibit multi-mode participation rather than diagonal isolation. |

## Figure 12: Temporal overlap heatmap
![POD temporal overlap heatmap](demo_visuals/pod_temporal_overlap.png)

> 🏡 OOrthogonality of right singular vectors $\langle v_m | v_n \rangle$, conforms to exact unitary requirements.

> 🔑 **Key Insights**
> 1. **Unitary property:** Diagonals equal $1.00$ and off-diagonals equal $\pm 0.00$.
> 2. **SVD Consistency:** Confirms numerical precision of the underlying SVD algorithm.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description**                                                                                  | **Explanation**                                                                                |
> | :---------- | :--------------- |:-------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------|
> | ✔️ | Identity deviation | Off-diagonal deviation from standard identity. | Off-diagonals are identically $0.00$, fully passing unitary criteria. | 

## Figure 13: Temporal cross-overlap
![POD temporal overlap heatmap](demo_visuals/pod_temporal_cross_overlap.png)

> 🏡 Absolute temporal coefficients $|V_{nk}| = |\langle \mathbf{e}_n | v_k \rangle|$ reveal modal mixing across snapshot states.

> 🔑 **Key Insights**
> 1. **Cross-State Participation:** Off-diagonal magnitudes reach $0.47$ ($n=0, k=1$) and $0.46$ ($n=1, k=0$).
> 2. **Partial State Isolation:** State $2$ maintains strong modal dominance with $k=2$ ($0.98$).

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | Spread dominance | Multiple temporal modes project onto single state. | States $0$ and $1$ exhibit shared weight distribution across modes $0$ and $1$. |
> 