# Figure Analysis (Windows Computer)

## Figure 1: Training Curves
![Training-curve panel](demo_visuals/training_curves.png)

> 🏡 The optimizer exhibits three distinct regime transitions before settling on a stable plateau.

> 🔑 **Key Insights**
> 1. **Spike 1 ($\approx$ 0-50 epochs)** - Expected transient while the network adjusts from random initial weights.
> 2. **Spike 2 ($\approx$ 800 epochs)** - Discovery of a higher-curvature potential: smoothness and total loss spike, physics and data terms rise only moderately.
> 3. **Spike 3 ($\approx$ 2000 epoochs)** - Order-of-magnitude jump in the smootheness term propagates into the physics loss; a new plateau follows with lower smoothness fidelity, but improved data fit.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> | ❌ | High final loss | Optimizer stalls in a local minimum. | Total loss remains greater than 1e-1 at epoch 6000. |
> | ✔️ | Oscillation avoided | Unbalanced loss weights can cause loss terms to oscillate. | Curves converge monotonically after Spike 3. |
> | ❌ | Physics collapse | Data loss decreases, while TISE residual increases. | Indicates operator inconsistency. |
> | ❌ | Over-regularization | Smoothness term dominates, spectrum becomes innacurate. | Post-Spike 3 plateau shows $\lambda_\text{smooth}$ is much greater than others. | 

## Figure 2: Learned vs. true potential
![Learned potentials](demo_visuals/learned_potential.png)

> 🏡 The learned potential $V_\theta(x)$ (sigmoidal) differs markedly from the harmonic ground truth $V(x)=\tfrac12 x^2$.

> 🔑 **Key Insights**
> 1. Central regions of the learned eigenfunctions (Fig. 3) and densities (Fig. 5) match the ground truth far better than the tails.
> 2. The model learns only the portion of $H_\theta$ required to reproduce high-probability regions, exposing the inverse problem's under-determinism.

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode**              | **Description**                                                                          | **Explanation**                                                    |
> |:------------|:------------------------------|:-----------------------------------------------------------------------------------------|:-------------------------------------------------------------------|
> | ❌           | Gemoetric mismatch       | Learned $V_\theta$ shape incompatible with true quadratic.                               | Central well too narrow; tails saturate at $V_\theta \approx \pm 12 $. |
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

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 7: Spatial overlap heatmap
![POD singular values](demo_visuals/pod_singular_values.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 8: POD spatial modes
![POD spatial modes](demo_visuals/pod_modes.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 


## Figure 9: Cross-overlap heatmap (POD vs. learned)
![Cross overlap: POD vs. learned](demo_visuals/cross_overlap_heatmap.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 10: POD Eigen-Alignment
![POD Eigen-Alignment](demo_visuals/pod_eigen_alignment.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 11: POD Temporal Modes
![POD temporal modes](demo_visuals/pod_temporal_modes.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 12: Temporal overlap heatmap
![POD temporal overlap heatmap](demo_visuals/pod_temporal_overlap.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 

## Figure 13: Temporal cross-overlap
![POD temporal overlap heatmap](demo_visuals/pod_temporal_cross_overlap.png)

> 🏡

> 🔑 **Key Insights**
> 1. 

> ❌ **Failure Modes**
> 
> | **Verdict** | **Failure Mode** | **Description** | **Explanation** |
> | :---------- | :--------------- | :-------------- | :-------------- |
> 