# Figure Analysis (Windows Computer)


## Figure 1
![training_curves](demo_visuals/training_curves.png)
> 🏡 The optimizer undergoes multiple regime transitions before setting into a stable solution.

> 🔑 **Key Insights**
> 1. **Spike 1 (~0-50 epochs):** To be expected if it is initial transient behavior (recall the network begins with random loss weights).
> 2. **Spike 2 (~800 epochs):** Likely the model discovered a new potential configuration that dramatically increased curvature. Evidence comes from
>    - smoothness loss curve spikes
>    - total loss curve spikes
>    - relatively moderate increase in physics and data-mismatch losses
> 3. **Spike 3 (~2000 epochs)**: Smoothness term spikes many orders of magnitude, then:
>    - physics loss curve increases
>    - total loss increases
>    - all curves settle on a different plateau than before the spike
>    This suggests the optimizer discovered a qualitatively different potential that better matched the observed wavefuncitons while sacrificing smoothness and physical fidelity.

> ❌ **Failure Modes**
> 1. **High final loss**: Optimization stalled in a local medium.
> 2. **Oscillation**: Unbalanced loss weights causing the optimizer to swing back and forth between physics and data.
> 3. **Physics Collapse**: Data losses decrease while the TISE loss remains high. This indicates that the model reproduces observed states, but fails to learn a physically consistent operator.
> 4. **Over-Regularization**: Smoothness penalties dominate optimization, producing physically smooth but spectrally inaccurate potentials.


## Figure 2
![learned_potential.png](demo_visuals/learned_potential.png)
> 🏡 The learned potential $V_\theta(x)$ differs significantly from the ground truth $V(x)=\frac{1}{2}x^2$. However, $V_\theta(x)$ gives a $H_\theta$ with comparable spectra to the ground truth spectra for the harmonic oscillator.  

> 🔑**Key Insights**
> 1. The central regions of the learned eigenfunctions (Figure 3) and associated densities (Figure 5) match the ground truth remarkably well compared to the peripheral regions.
> 2. Errors are concentrated in the peripheral regions / tails of learned curves in Figures 3 and 5.
> 3. The first two observations suggest that the model primarily learned the portion of $H_\theta$ needed to reproduce the observed probability density in regions where the data contain large amplitude. To put it simply, the inverse problem appears to be underdetermined.

> ❌ **Failure Modes**
> 
> 

## Figure 3
![learned_wavefunctions.png](demo_visuals/learned_wavefunctions.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 


## Figure 4
![learned_energies.png](demo_visuals/learned_energies.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 


## Figure 5
![density.png](demo_visuals/density.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 


## Figure 6
![pod_singular_values.png](demo_visuals/pod_singular_values.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 7
![overlap_heatmap.png](demo_visuals/overlap_heatmap.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 8
![pod_modes.png](demo_visuals/pod_modes.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 9
![cross_overlap_heatmap.png](demo_visuals/cross_overlap_heatmap.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 10
![pod_temporal_modes.png](demo_visuals/pod_temporal_modes.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 11
![pod_temporal_overlap.png](demo_visuals/pod_temporal_overlap.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 

## Figure 12
![pod_temporal_cross_overlap.png](demo_visuals/pod_temporal_cross_overlap.png)
> 🏡 

> 🔑**Key Insights**
> 

> ❌ **Failure Modes**
> 
> 