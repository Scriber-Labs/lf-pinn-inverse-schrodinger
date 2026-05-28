# 🐉 Figure Analysis (Windows)
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

### 🧿  Numerical Methods
- [ ] Is the following matrix description of $H_\theta$ consistent with the figures generated in demo.ipynb and cli scripts?

    $$ H_\theta = -\frac{1}{2} D_{xx} + \text{diag}(V_\theta(x)) \, ❓$$

  - [ ] If is is accurate, can you write out the expressoins for $D_{xx}$ and $\text{diag}(V_\theta(x))$ ?
  - [ ] Please refer the @assets/mermaid-diagrms/architecture_final.mmd ?
- [ ] Where is the best figure to mention ill-conditioning?

--

## 🎭 **Setting the Stage:**
- The PIML learns $N=3$ orthonormalized eigenfunctions $\hat{\psi}_n^\theta$ where $n\in\{0,1,\dots,N-1\}$.
- [ ] `<<Enter description of learned $E_n^\theta$>>`
- [ ] enter other important variables

## Figure 1
![training_curves](demo_visuals/training_curves.png)
> 🏡**Take-Home Message:** 
> 
>

> 🔑 **Key Ideas:**
> 1. The physics residual $R_n = H_\theta \psi_n^\theta -E_n^\theta\psi_n^\theta $ where $$ H_\theta \hat{\psi}_n^\theta \approx E_n^\theta \hat{\psi}_n^\theta $$ gives $$\mathcal{L}_\text{TISE} = \sum_{n=0}^{N-1}\| R_n(x_j)  \|^2 \quad \text{⚠️ this needs to be revised and sanity checked by myself}$$
> 2. $\mathcal{L}_\text{order} = ?$ 
> 3. The smoothness term $$ \mathcal{L}_\text{smooth} = \Bigg\langle\frac{\|V_\theta''(x_j) \|^2}{\epsilon + \| V_\theta(x_j) \|^2}\Bigg\rangle $$
> 4. The data mismatch term (pink curve) is pointwise supervised $\Rightarrow$ $\mathcal{L}_\text{data}$ directly supervises $\rho_n^\text{obs}$ and $E_n^\theta$:
>    
>    $$ \mathcal{L}_\text{data} = \frac{1}{N}\sum_{n=0}^{N-1}{\Big(E_n^\theta - E_n^\text{obs}\Big)^2}+\frac{1}{N}\sum_{n=0}^{N}{\frac{1}{M}\sum_{j=1}^{M}\Big(|\hat{\psi}_n^\theta(x_j)|^2-\rho_n^\text{obs}(x_j)\Big)^2} $$
>    
>    where $N$ is the number of learned eigenfunctions and the

> ❌**Failure Modes:**
> 
> - 
> - 

---
## Figure 2
![learned_potential](demo_visuals/learned_potential.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 

> ❌**Failure Modes:**
> 
> - 
> 

---
## Figure 3
![learned_wavefunctions](demo_visuals/learned_wavefunctions.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 
> ---
## Figure 4
![learned_energies](demo_visuals/learned_energies.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 
> 
> ---
## Figure 5
![density](demo_visuals/density.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> ---
## Figure 6
![pod_singular_values](demo_visuals/pod_singular_values.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> - [ ] Discuss failure modes related to degenerate singular values.
> - 
> 
> ---
## Figure 7
![pod_modes](demo_visuals/pod_modes.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> -
> -

---
## Figure 8
![overlap_heatmap](demo_visuals/overlap_heatmap.png)
> 🏡**Take-Home Message:** 
> Identity matrix tells us $\hat\psi_n^\theta$ were successfully orthonormalized.

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 

---
## Figure 9
![cross_overlap_heatmap](demo_visuals/cross_overlap_heatmap.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 

---
## Figure 10
![pod_temporal_modes](demo_visuals/pod_temporal_modes.png)
> 🏡**Take-Home Message:** 
> 
> 

> 📒 **Terminology**
> 
> - **'temporal' modes** $V_k$
>     - ⚠️ Note 'temporal' in a physical sense, but in a mathematical sense. Need to find a 'go-to' replacement for Scriber Labs

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 

---
## Figure 11
![pod_temporal_overlap](demo_visuals/pod_temporal_overlap.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 


---
## Figure 12
![pod_temporal_cross_overlap](demo_visuals/pod_temporal_cross_overlap.png)
> 🏡**Take-Home Message:** 
> 

> 🔑 **Key Ideas:**
> 
> -

> ❌**Failure Modes:**
> 
> - 
> 