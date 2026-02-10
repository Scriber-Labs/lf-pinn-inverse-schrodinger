# Figure Analysis
## Figure 1 - Training Curves
![Training Curves](demo_visuals/training_curves.png)

> 🏡 *"The inverse problem stabilizes under competing physical and data-driven objectives."*

---

## Figure 2 - Learned Potential $V_\theta(x)$ vs. Ground Truth Potential $V(x)$ (Harmonic Oscillator)
![Potential Functions](demo_visuals/learned_potential.png)

### 🏡 Take-Home Messages
- **Smoothness prior effect** pulls $V_\theta(x)$ towards a low-curvature shape.
- **Identifiability limits** (❓) 
- **Bias vs. variance tradeoff:** 
  - If $\lambda_\text{smooth}$ is too strong, flattening (bias) occurs.
  - If $\lambda_\text{smooth}$ is too weak, noisy wiggles (wiggles) manifest.

### ✖️ Failure Modes
- **Flattening** (bias) occurs if $\lambda_\text{smooth}$ is too strong.
- **Over-smoothing** (❓is this the same thing as flattening, or is it more general/ something different?❓)
- **Boundary artifacts** are more likely to manifest due to the model being less constricted near the boundaries.

### 🔮 Future Projects
- Dynamic weighting of the loss terms
  - 📝 For this project, the loss weights are assigned to a static `dict`. 
  - Ideas
    - Focus on data first, then enforce physics loss.
    - Prevent over-smoothing early during training.
    - **Adaptive balancing:** Compute the magnitude of each loss term every epoch and scale the $\lambda$'s so that all terms contribute roughly the same amount.
    - **Bayesian/ probabilistic sampling:** Treat each $\lambda$ as a learnable hyperparameter and update it with gradient descent.
  - Tips
    - Start with a static `dict` for the first pass. This will provide a baseline to compare against.
    - Add a **scheduler** only if you see a failure mode symptom.
    
      | **Symptom** | **Remedy**                                                       |
      | ----------- |------------------------------------------------------------------|
      | Physics residual stalls while the data-fit continues to improve. | Increase $\lambda_\text{physics}$.                               |
      | Flattening failure mode | Decrease $\lambda_\text{smooth}$ or slow down its schedule.      |
       | Wildly oscillating loss curves | Use a **smooth ramp** for all $\lambda$'s to stabilize training. | 
    
    - Log the effective $\lambda$ values each epoch and plot them alongside the training curves. This will help reveal whether a scheduler helped or hindered learning.
---

## Figure 3 - Learned Wavefunctions $\psi_n^\theta(x)$ vs. Ground Truth Wavefunctions $\psi_n(x)$ (Harmonic Oscillator)


### 🏡 Take-Home Messages
- **Phase ambiguity**
- **Shape consistency**
- **Node structure**
