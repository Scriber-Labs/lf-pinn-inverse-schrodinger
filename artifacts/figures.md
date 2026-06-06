# Figure Analysis (Windows Computer)


## Figure 1 : Training Curves
![training_curves](demo_visuals/training_curves.png)
> 🏡 The optimizer undergoes multiple regime transitions before setting into a stable solution.

> 🔑 **Key Insights**
> 1. **Spike 1 (~0-50 epochs):** To be expected if it is initial transient behavior (recall the network begins with random loss weights).
> 2. **Spike 2 (~800 epochs):** Likely the model discovered a new potential configuration that dramatically increased curvature. Evidence comes from:
>    - Smoothness loss curve spikes.
>    - Total loss curve spikes.
>    - Relatively moderate increase in physics and data-mismatch losses.
> 3. **Spike 3 (~2000 epochs)**: Smoothness term spikes many orders of magnitude, then:
>    - Physics loss curve increases.
>    - Total loss increases.
>    - All curves settle on a different plateau than before the spike.
>    
>    This suggests the optimizer discovered a qualitatively different potential that better matched the observed wavefuncitons while sacrificing smoothness and physical fidelity.

> ❌ **Failure Modes**
> 1. **High final loss**: Optimization stalled in a local medium.
> 2. **Oscillation**: Unbalanced loss weights causing the optimizer to swing back and forth between physics and data.
> 3. **Physics Collapse**: Data losses decrease while the TISE loss remains high. This indicates that the model reproduces observed states, but fails to learn a physically consistent operator.
> 4. **Over-Regularization**: Smoothness penalties dominate optimization, producing physically smooth but spectrally inaccurate potentials.


## Figure 2
![learned_potential.png](demo_visuals/learned_potential.png)
> 🏡 The learned potential $V_\theta(x)$ (a sigmoidal-like shapae) differs significantly from the ground truth $V(x)=\frac{1}{2}x^2$.

> 🔑**Key Insights**
> 1. The central regions of the learned eigenfunctions (Figure 3) and associated densities (Figure 5) match the ground truth remarkably well compared to the peripheral regions.
> 2. Errors are concentrated in the peripheral regions / tails of learned curves in Figures 3 and 5.
> 3. The first two observations suggest that the model primarily learned the portion of $H_\theta$ needed to reproduce the observed probability density in regions where the data contain large amplitude. To put it simply, the inverse problem appears to be underdetermined.

> ❌ **Failure Modes**
> - Spectrally consistent, but geometrically incorrect operators.
> - Boundary under-constraint.

## Figure 3 : Learned Wavefunctions
![learned_wavefunctions.png](demo_visuals/learned_wavefunctions.png)
> 🏡 The learned eigenfunctions $\psi_n^\theta(x)$ (solid lines) are compared against the analytic Hermite-Gaussian ground truth (dashed lines).

> 🔑**Key Insights**
> 1. **Phase Matching**: The model correctly captures the nodal structure (number of zeros) for the first few states, which is critical for operator consistency.
> 2. **Central Accuracy**: Similar to the potential, the wavefunctions are most accurate in the central region where the probability density is highest.
> 3. **Tail Divergence**: In the peripheral regions ($|x| > 2$), the learned wavefunctions deviate from the ground truth, reflecting the under-determined nature of the inverse problem in low-amplitude regions.

> ❌ **Failure Modes**
> 1. **Nodal Miss-count**: The model fails to learn the correct number of nodes, indicating a failure to capture the correct energy level.
> 2. **Sign/Phase Flip**: While we align signs for visualization, an unaligned model might show inverted parity.
> 3. **Spurious Oscillations**: High-frequency ripples in the tails, often caused by over-fitting to noise or lack of smoothness in $V_\theta$.


## Figure 4 : Ground Truth vs. Learned Energy Eigenvalues
![learned_energies.png](demo_visuals/learned_energies.png)
> 🏡 The learned energy spectrum $\{E_n^\theta\}$ matches the observed energy values and closely follows the ground truth spectrum for the harmonic oscillator, $E_n = n + 0.5$.

> 🔑**Key Insights**
> 1. The recovered spectrum remains correctly ordered throughout the final solution, indicating that the $\mathcal{L}_\text{order}$ was effective.
> 2. The model successfully generalized a consistent spectrum, despite training on noisy data.

> ❌ **Failure Mode**: The learned energies match the observed spectrum while the reconstructed potential differs substantially from the true potential. This indicates successful spectral fitting but incomplete recovery of the underlying operator.

## Figure 5 : Learned Density vs. Observed Data
![density.png](demo_visuals/density.png)
> 🏡 Comparison between the probability densities $\rho_n^\theta = |\psi_n^\theta|^2$ and the synthetic noisy observations.

> 🔑**Key Insights**
> 1. **Noise Filtering**: The PINN acts as a physics-informed smoother, capturing the underlying density profile despite the 2% Gaussian noise in the observations.
> 2. **Data Dominance**: The close match here, even when the potential (Figure 2) is incorrect, confirms that the $\mathcal{L}_\text{data}$ term is easily satisfied.

> ❌ **Failure Modes**
> 1. **Peak Flattening**: Oversmoothing (high $\lambda_\text{smooth}$) can lead to lowered peaks in the density.
> 2. **Mode Merging**: If the energy ordering fails, multiple learned states might collapse onto the same observed density profile.


## Figure 6 : POD Singular Values
![pod_singular_values.png](demo_visuals/pod_singular_values.png)
> 🏡 The singular value spectrum from the Proper Orthogonal Decomposition of the learned wavefunction matrix.

> 🔑**Key Insights**
> 1. **Rank Efficiency**: A sharp decay in singular values suggests that the learned states are well-represented by a low-dimensional basis.
> 2. **Basis Conditioning**: The magnitude of the first $N$ singular values relative to the rest indicates how much of the learned "physics" is concentrated in the intended modes.

> ❌ **Failure Modes**
> 1. **Flat Spectrum**: Indicates that the learned states are linearly independent but not necessarily physically structured (e.g., random noise).
> 2. **Slow Decay**: Suggests the model is learning redundant or highly correlated information across modes.


## Figure 7 : Spatial Overlap Heatmap
![overlap_heatmap.png](demo_visuals/overlap_heatmap.png)
> 🏡 Matrix of inner products $\langle \psi_i^\theta | \psi_j^\theta \rangle$ showing the orthogonality of the learned eigenfunctions.

> 🔑**Key Insights**
> 1. **Orthogonality**: The diagonal dominance (values near 1.0) and near-zero off-diagonal elements indicate that the model has successfully learned an orthogonal basis, as required by the TISE.
> 2. **Basis Consistency**: High off-diagonal values would suggest the $\mathcal{L}_\text{physics}$ was insufficient to enforce the Hermitian operator's property of orthogonal eigenstates.

> ❌ **Failure Modes**
> 1. **Non-Orthogonality**: Significant off-diagonal components ($> 0.1$) indicate the learned operator is not Hermitian or the training hasn't converged.


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