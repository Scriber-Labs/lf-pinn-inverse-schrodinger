# Figure Analysis (Windows Computer)

## Figure 1 : Training Curves
![training_curves](demo_visuals/training_curves.png)
> 🏡 The optimizer exhibits three distinct regime transitions before settling into a stable solution.

> 🔑 **Key Insights**
> 1. **Spike 1 (~0-50 epochs):** Expected transient while the network adjusts from random initial weights.
> 2. **Spike 2 (~800 epochs):** Indicative of the model discovering a new potential configuration with larger curvature. This is evidenced by concurrent spikes in the smoothness and total-loss curves while physics and data-mismatch losses rise only moderately.
> 3. **Spike 3 (~2000 epochs):** A multi-order-of-magnitude jump in the smoothness term leads to higher physics loss; all curves then settle on a different plateau, suggesting the optimizer found a qualitatively different potential that fits the observed wavefunctions better at the expense of smoothness and physical fidelity.

> ❌ **Failure Modes**
> 
>  | Failure Mode | Description | Pass / Fail | Explanation |
>  |:-------------|:------------|:------------|:------------|
>  | **High final loss** | The optimizer fails to converge to a global minimum, settling in a suboptimal solution. | ❌           | Optimization stalled a local medium. | 
>  | **Oscillation** | Unbalanced loss weights cause the optimizer to oscillate between two loss terms | ✔️          | The optimizer successfully balances the physics and data terms, avoiding divergence. |
>  | **Physics Collapse** | Data losses decrease while the TISE loss remains high, indicating that the model reproduces observed spectral features but not a consistent operator. | ❌          | The model fails to capture the underlying physics accurately. |
>  | **Over-Regularization** | Smoothness penalties dominate optimization, yielding physically smooth but spectrally inaccurate potentials. | ❌          | The model prioritizes smoothness over physical fidelity. |

## Figure 2
![learned_potential.png](demo_visuals/learned_potential.png)
> 🏡 The learned potential $V_\theta(x)$ (a sigmoidal-like shape) differs markedly from the ground-truth 1-D harmonic potential $V(x)=\frac{1}{2}x^2$.

> 🔑**Key Insights**
> 1. Central regions of the learned eigenfunctions (Fig. 3) and associated densities (Fig. 5) match the ground truth better than the tails.
> 2. The model appears to learn only the portion of $H_\theta$ needed to preproduce high-probability regions, revealing the inverse problem's under-determined nature.

> ❌ **Failure Modes**
> 
>  | Failure Mode | Description | Pass / Fail | Explanation |
>  |:-------------|:------------|:------------|:------------|
>  |  | Geometrically incorrect learned potential function. | ❌           |  |
>  | **Boundary under-constraint** | |          | |


## Figure 3 : Learned Wavefunctions
![learned_wavefunctions.png](demo_visuals/learned_wavefunctions.png)
> 🏡 Learned eigenfunction $\psi_n^\theta(x)$ (solid) errors concentrate in the peripheral regions, hinting that low-amplitude data under-constrain the inverse problem.

> 🔑**Key Insights**
> 1. **Phase matching:** Correct nodal structure confirms energy-level ordering.
> 2. **Central accuracy:** Highest fidelity occurs where $|\psi_n^\theta|^2$ is largest.
> 3. **Tail divergence:** For ($|x| > 2$), deviations reflect under-determination in low-density regions.

> ❌ **Failure Modes**
> 
>  | Failure Mode          | Description                                                                                       | Pass / Fail | Explanation                                                                                                                                              |
>  |:----------------------|:--------------------------------------------------------------------------------------------------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
>  | **Nodal mis-count**   | Wrong number of zeros indicates an energy-ordering failure.                                       | ❌           | The learned eigenfunctions approximate the correct number of nodal regions in central regions, but begin to overproduce nodes in the peripheral regions. |
>  | **Sign / phase flip** | An un-aligned solution may invert parity.                                                         | ✔️          | The learned eigenfunctions have parity that corresponds with the ground truth.                                                                           |
>  | **Surious Oscillations** | High-frequency ripples in the tails arise from over-fitting or insufficient $V_\theta$ smoothness | ❌           | Additional diagnostics are needed for further analysis.                                                                                                  |


## Figure 4 : Ground Truth vs. Learned Energy Eigenvalues
![learned_energies.png](demo_visuals/learned_energies.png)
> 🏡 The energies $\{E_n^\theta\}$ align with observations and follow the harmonic spectrum $E_n = n + 0.5$.

> 🔑**Key Insights**
> 1. Spectrum remains correctly ordered, suggesting $\mathcal{L}_\text{order}$ is effective.
> 2. The model successfully generalized a consistent spectrum, despite training on Gaussian noisy data.

> ❌ **Failure Mode** 
>
>  | Failure Mode                                                  | Description                                                      | Pass / Fail | Explanation                                                                                                                                                                                                                      |
>  |:--------------------------------------------------------------|:-----------------------------------------------------------------|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
>  | **Correct spectral fit;**<br/>**Incorrect operator geometry** | Spectral fit survives even when reconstructed operator is wrong. | ❌           | The learned energies match the observed spectrum while the reconstructed potential differs substantially from the true potential. This indicates successful spectral fitting but incomplete recovery of the underlying operator. |


## Figure 5 : Learned Density vs. Observed Data
![density.png](demo_visuals/density.png)
> 🏡 Comparison of learned probability densities $\rho_n^\theta = |\psi_n^\theta|^2$ with synthetic noisy observations ($2\, \%$ Gaussian noise).

> 🔑**Key Insights**
> 1. **Noise Filtering**: The PINN acts as a physics-informed smoother.
> 2. **Data Dominance**: The close match here, even when the potential (Figure 2) is incorrect, confirms that the $\mathcal{L}_\text{data}$ term is easily minimized.

> ❌ **Failure Modes**
> 
>  | Failure Mode                                                                                                                   | Description                                                                           | Pass / Fail | Explanation                                                                                                                                                                                                                                |
>  |:-------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------|:------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
>  | **Peak Flattening**                                                                                                            | Oversmoohting (high $\lambda_\text{smooth}$) can lead to lowered peaks in the desnity | ✔️          | The learned density peaks match the observed density peaks.                                                                                                                                                                                |
>  | **Mode Merging**                                                                                                               | If energy ordering fails, multiple states can collapse onto one density profile.      |  ✔️         | Learned density matches the observed peaks approximetly well (particularly near center regions). This is consistent with correclty orered energy eigenvalues in Fig. 4. |

---

## Figure 6 : POD Singular Values
![pod_singular_values.png](demo_visuals/pod_singular_values.png)
> 🏡 The singular value spectrum from the POD of the learned wavefunction matrix.

> 🔑**Key Insights**
> 1. **Rank Efficiency**: Rapid decay implies a low-dimensional representation.
> 2. **Basis Conditioning**: Relative magnitudes of the first three singular values show how much "physics" resides in intended modes. 

> ❌ **Failure Modes**
> 1. **Flat Spectrum**: States are independent but lack physical structure.
> 2. **Slow Decay**: Redundant or highly correlated modes.


## Figure 7 : Spatial Overlap Heatmap
![overlap_heatmap.png](demo_visuals/overlap_heatmap.png)
> 🏡 Matrix of inner products $\langle \psi_i^\theta | \psi_j^\theta \rangle$ showing the orthogonality of the learned eigenfunctions.

> 🔑**Key Insights**
> 1. **Orthogonality**: Near-unity diagonal and near-zero off-diagonals confirm Hermitian-like behavior of learned eigenfunctions.
> 2. **Basis Consistency**: Large off-diagonals would expose insufficient $\mathcal{L}_\text{physics}$.

> ❌ **Failure Modes**
> 1. **Non-Orthogonality**: Significant off-diagonal components ($> 0.1$) indicate non-orthogonality or incomplete convergence.


## Figure 8 : POD Spatial Modes
![pod_modes.png](demo_visuals/pod_modes.png)
> 🏡 The first three spatial POD modes extracted from the learned wavefunction ensemble.

> 🔑**Key Insights**
> 1. **Geometric Structure**: POD modes represent the "optimal" basis for the learned data. Their similarity to the learned wavefunctions (Figure 3) indicates a stable basis.
> 2. **Feature Extraction**: These modes highlight the most persistent spatial features across the learned spectrum.

> ❌ **Failure Modes**
> 1. **Mode Mixing**: POD modes that do not resemble any physical eigenfunctions suggest the learned ensemble lacks physical coherence.


## Figure 9 : Cross-Overlap Heatmap (POD vs. Learned)
![cross_overlap_heatmap.png](demo_visuals/cross_overlap_heatmap.png)
> 🏡 Projection of the learned wavefunctions onto the POD basis $\langle \psi_n^\theta | u_k \rangle$.

> 🔑**Key Insights**
> 1. **Alignment**: Ideally, this should be an identity matrix if each learned state corresponds exactly to one POD mode.
> 2. **Energy Concentration**: Shows how the energy of the learned states is distributed across the POD basis.

> ❌ **Failure Modes**
> 1. **Distributed Overlap**: If a single state projects onto many POD modes, it indicates a lack of clear modal structure in the learned solution.


## Figure 10 : POD-Eigen Alignment
![pod_eigen_alignment.png](demo_visuals/pod_eigen_alignment.png)
> 🏡 Cross-overlap between physical POD modes and ground truth eigenfunctions $\langle u_k | \psi_n^\text{true} \rangle$.

> 🔑**Key Insights**
> 1. **Absolute Consistency**: This is the "gold standard" diagnostic. High diagonal values indicate that the POD modes (derived from the learned model) have successfully recovered the true physical basis.
> 2. **Spectral Recovery**: Success here confirms that the model captured the correct operator structure, even if the potential $V_\theta$ looks different.



## Figure 11 : POD Temporal Modes
![pod_temporal_modes.png](demo_visuals/pod_temporal_modes.png)
> 🏡 Visualization of the $V$ matrix from the SVD ($\Psi = U S V^T$), representing the "temporal" (state-index) coefficients.

> 🔑**Key Insights**
> 1. **Coefficient Distribution**: Shows how each physical POD mode contributes to each learned state index.

> ❌ **Failure Modes**
> 1. **Incoherent Coefficients**: Random-looking coefficients suggest the POD basis is not effectively capturing the state-wise variations.


## Figure 12 : Temporal Overlap Heatmap
![pod_temporal_overlap.png](demo_visuals/pod_temporal_overlap.png)
> 🏡 Orthogonality check for the temporal coefficients.

> 🔑**Key Insights**
> 1. **Unitary Property**: Since $V$ is a unitary matrix from SVD, this should strictly be the identity matrix. It serves as a diagnostic for the SVD implementation and basis conditioning.

> ❌ **Failure Modes**
> 1. **Identity Deviations**: Indicates numerical instability or errors in the POD decomposition pipeline.


## Figure 13 : Temporal Cross-Overlap
![pod_temporal_cross_overlap.png](demo_visuals/pod_temporal_cross_overlap.png)
> 🏡 Absolute values of the temporal coefficients $|V_{nk}|$.

> 🔑**Key Insights**
> 1. **Modal Dominance**: Highlights which POD modes are the primary contributors to which learned states.

> ❌ **Failure Modes**
> 1. **Spread Dominance**: No clear diagonal or sparse structure indicates a lack of correspondence between POD modes and learned states.


<!-- ## Figure 14 : Hilbert Space Phase Portrait
![hilbert_portrait.png](demo_visuals/hilbert_portrait.png)
> 🏡 3D projection of learned wavefunctions into the subspace spanned by the first three true eigenfunctions.

> 🔑**Key Insights**
> 1. **Trajectory in Hilbert Space**: Visualizes how the learned states "cluster" around the true eigenstates.
> 2. **Global Basis Conditioning**: A clear separation of points in this space indicates a well-conditioned, non-degenerate learned basis.

> ❌ **Failure Modes**
> 1. **Clustering/Collapse**: Multiple points collapsing to the same region suggests the model is failing to distinguish between different energy levels.


## Figure 15 : Spectral Energy Cascade
![spectral_cascade.png](demo_visuals/spectral_cascade.png)
> 🏡 Comparison between POD singular values and learned energy eigenvalues.

> 🔑**Key Insights**
> 1. **Spectral Correlation**: Parallel decay of singular values and energy levels suggests that the POD basis is physically well-aligned with the Hamiltonian's spectrum.
> 2. **Energy Concentration**: Most of the system's "energy" is concentrated in the first few modes, matching the expected low-energy behavior.

## Figure 16 : POD Partition Function Spectrum
![partition_spectrum.png](demo_visuals/partition_spectrum.png)
> 🏡 Thermodynamic interpretation of the POD spectrum as a Boltzmann-like distribution.

> 🔑**Key Insights**
> 1. **Entropy Metric**: The entropy value quantifies the "disorder" or complexity of the learned solution. Lower entropy indicates a more structured, physically coherent basis.
> 2. **Effective Energies**: The linear relationship in effective energy highlights the hierarchical nature of the learned modes.

-->


## Conclusions
- **Spectral Recovery**: Figures 4 and 13 demonstrate that the model accurately recovers the energy eigenvalues and the core eigenfunction basis (via POD alignment).
- **Potential Identifiability**: Figure 2 highlights that the inverse problem is under-determined; many potentials can produce the same low-energy spectral data, especially in regions with low sampling density.
- **Operator Consistency**: Figures 7 and 11 confirm that the learned operator maintains Hermitian-like properties (orthogonality of states).
- **POD Diagnostics**: The POD pipeline (Figures 6-12) provides a robust framework for evaluating the geometric structure and basis conditioning of the PINN solution, independent of the explicit potential form.