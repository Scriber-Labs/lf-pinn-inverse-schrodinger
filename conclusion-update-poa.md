# Plan of Action: Final Major Architecture & Diagnostic Update
**Reference Citation:** `salmanogli2026QNN` (`references.bib`), Brunton & Kutz (2022), Raissi et al. (2019), Hestenes (1993).

---

## Part 1: Comprehensive Inventory of Architecture Nodes, Diagrams, Plots, Colormaps, & Metrics

This inventory maps every functional node across the physics-informed neural network (PINN), proper orthogonal decomposition (POD), and quantum Hamiltonian learning architecture to its corresponding diagnostic plots, heatmaps, mathematical forms, internal metrics, external metrics, and exposed failure modes.

```text
+--------------------------------------------------------------------------------------------------------------------------+
|                                              PIML & POD DIAGNOSTIC FLOW MAP                                              |
+--------------------------------------------------------------------------------------------------------------------------+
|  [0. Physical System] TISE Formulation: (-1/2 D2 + V) psi = E psi                                                        |
|           |                                                                                                              |
|  [1. Grid & Data]     Uniform Grid x in [-5, 5], dx, w_trap  --> Observed Data: rho_n^obs, E_n^obs                       |
|           |                                                                                                              |
|  [2. Neural Ansatz]   MLP_V -> V_theta | MLPs -> psi_raw -> Orthonormalization (GS) -> psi_hat | Energy -> E_n^theta     |
|           |                                                                                                              |
|  [3. Residual / Loss] Physics Residual R_n(x), L_TISE, L_smooth, L_data, L_order, L_fidelity (Salmanogli 2026)           |
|           |                                                                                                              |
|  [4. Diagnostics]     Figs 1a-c (Losses) | Fig 2 (V_theta) | Fig 3 (psi_n) | Fig 4 (E_n) | Fig 5 (rho_n)                 |
|           |                                                                                                              |
|  [5. POD Analysis]    Psi_w = W^(1/2) Psi -> SVD: U Sigma V^T -> Physical Modes u_k^phys = W^(-1/2) U                  |
|           |                                                                                                              |
|  [6. Operator Proj.]  Discrete Hamiltonian H = -1/2 D_xx + diag(V) -> Galerkin: H_pod = U^* H U                          |
|           |                                                                                                              |
|  [7. Modal Heatmaps]  Fig 6 (Sigma_k) | Fig 7 (Overlap <psi|psi>) | Fig 8 (u_k vs psi) | Fig 9 (<u|psi_theta>)           |
|                       Fig 10 (<u|psi_true>) | Fig 11 (V_nk) | Fig 12 (<v_m|v_n>) | Fig 13 (|<e_n|v_k>|)                 |
|           |                                                                                                              |
|  [8. QNN Emulation]   Fig 14 (State Fidelity & Trace Distance) | Fig 15 (Hamiltonian Matrix Error)                       |
|                       Fig 16 (Perturbation & Excitation Stability Spectrum) [Salmanogli 2026]                           |
+--------------------------------------------------------------------------------------------------------------------------+
```

### Detailed Node-to-Diagram & Metric Mapping Table

| Diagram / Figure ID | Architecture Node(s) | Plot / Colormap Type | Mathematical Formulation | Internal Success Metric (Self-Consistency) | External Success Metric (Ground Truth) | Failure Mode(s) Exposed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 1a** | `loss`, `opt` | Multi-line semi-log convergence curve (Total, Physics, Data, Smoothness, Order) | $\mathcal{L}_{\text{total}} = \lambda_{\text{phys}}\mathcal{L}_{\text{phys}} + \lambda_{\text{data}}\mathcal{L}_{\text{data}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}} + \lambda_{\text{order}}\mathcal{L}_{\text{order}}$ | Monotonic loss decrease; asymptotic plateau $\mathcal{L} < 10^{-4}$; no gradient explosion | N/A (unsupervised loss dynamics) | Optimizer stall, stiffness imbalance ($\lambda_i$), loss oscillations |
| **Figures 1b–1c** | `opt`, `potential`, `gs` | Zoomed-in line plots of training transients | $\hat{V}_\theta(x, t), \hat{\psi}_n^\theta(x, t)$ at epochs $5, 782$ | Rapid recovery from initial random initialization; smooth curvature transitions | N/A | High-curvature potential bifurcation shock, transient instability |
| **Figure 2** | `potential` ($V_\theta$) | 1D Cartesian line plot ($V_\theta$ vs. $V_{\text{true}}$) | $V_\theta(x) = \text{MLP}_V(x)$ | Boundedness: $\min V_\theta > -\infty$; symmetry $V_\theta(x) \approx V_\theta(-x)$ | Potential RMSE: $\|V_\theta - V_{\text{true}}\|_{L^2} < 0.05$ | Under-determinism in low-density boundaries ($\rho_n \approx 0$) |
| **Figure 3** | `psinorm` ($\hat{\psi}_n$) | Multi-panel 1D curves (Learned vs. Analytic) | $\hat{\psi}_n^\theta(x) = \text{GramSchmidt}(\text{MLP}_{\psi}(x))$ | Nodal count: $\text{roots}(\hat{\psi}_n) = n$; Parity: $\hat{\psi}_n(-x) = (-1)^n \hat{\psi}_n(x)$ | Eigenfunction fidelity: $\|\hat{\psi}_n^\theta - \psi_n^{\text{true}}\|_{L^2} < 0.01$ | State swapping, nodal displacement, high-frequency spurious ripples in tails |
| **Figure 4** | `energy` ($E_n^\theta$) | Discrete scatter / stem plot ($E_n^\theta$ vs. $E_n^{\text{true}}$) | $E_n^\theta = \text{Softplus}(\mathbf{w}_E)_n$ | Spectral monotonicity: $E_0^\theta < E_1^\theta < E_2^\theta$; Gap: $\Delta E \approx \hbar \omega$ | Relative spectral error: $\max_n \|E_n^\theta - E_n^{\text{true}}\| / E_n^{\text{true}} < 0.01$ | Spectral collapse, DC offset drift ($E_n \to E_n + C$) |
| **Figure 5** | `psinorm`, `observed_data` | 1D probability density lines vs. noisy scatter | $\rho_n^\theta(x) = |\hat{\psi}_n^\theta(x)|^2 \text{ vs. } \rho_n^{\text{obs}}$ | Positivity: $\rho_n^\theta(x) \ge 0$; Unit integral: $\int \rho_n^\theta dx = 1.0 \pm 10^{-4}$ | Density MAE: $\|\rho_n^\theta - \rho_n^{\text{obs}}\|_{L^1} < 0.02$ | Asymmetric peak broadening, background noise fitting |
| **Figure 6** | `spec` ($\Sigma$) | Semi-log discrete spectrum bar/line plot | $\sigma_k$ from $\mathbf{\Psi}_w = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ | Spectral decay ratio: $\sigma_0 / \sigma_k > 10^3$ for $k \ge 3$; Cumulative energy: $\sum_{k=0}^{N-1} \sigma_k^2 / \|\mathbf{\Psi}_w\|_F^2 = 1.0$ | Rank capture: $\text{rank}(\mathbf{\Psi}_w) = N_{\text{states}}$ | Flat spectrum (noise dominance/over-parameterization), slow decay (under-fitting) |
| **Figure 7** | `normalization`, `gs` | Discrete 2D heatmap (`viridis` / `coolwarm`) | $S_{ij} = \langle \hat{\psi}_i^\theta, \hat{\psi}_j^\theta \rangle_W = (\hat{\boldsymbol{\psi}}_i^\theta)^T \mathbf{W} \hat{\boldsymbol{\psi}}_j^\theta$ | Orthonormality error: $\|\mathbf{S} - \mathbf{I}_N\|_F < 10^{-5}$ | N/A (internal Hilbert space structure) | Incomplete Gram-Schmidt orthogonalization, gradient cancellation |
| **Figure 8** | `spatial_modes` ($u_k^{\text{phys}}$) | 1D spatial curves ($u_k^{\text{phys}}$ vs. $\psi_k^\theta$ vs. $\psi_k^{\text{true}}$) | $\mathbf{u}_k^{\text{phys}} = \mathbf{W}^{-1/2} \mathbf{u}_k$ | Structural regularity: spatial continuity $\|\nabla^2 u_k^{\text{phys}}\|_2 < \infty$ | Mode alignment: $|\langle u_k^{\text{phys}}, \psi_k^{\text{true}} \rangle_W| > 0.999$ | High-frequency noise amplification from $\mathbf{W}^{-1/2}$ weighting |
| **Figure 9** | `overlaps` ($C_{kj}$) | $N \times N$ matrix heatmap (`magma` / `viridis`) | $C_{kj} = \langle u_k^{\text{phys}}, \hat{\psi}_j^\theta \rangle_W = \mathbf{u}_k^T \mathbf{W}^{1/2} \hat{\boldsymbol{\psi}}_j^\theta$ | Diagonal dominance: $|C_{kk}| / \sum_j |C_{kj}| > 0.99$ | Alignment consistency across training epochs | Off-diagonal modal dispersion, rotation of empirical eigenbasis |
| **Figure 10** | `spatial_branch`, `sanity` | $N \times N$ matrix heatmap (`coolwarm` / `bwr`) | $A_{kk} = \langle u_k^{\text{phys}}, \psi_k^{\text{true}} \rangle_W$ | N/A (external assessment) | Absolute physical alignment: $|A_{kk}| > 0.995$ | Phase flip ($\pm 1$ sign ambiguity), unphysical mode permutation |
| **Figure 11** | `temporal_branch` ($V_{nk}$) | $N \times N$ composition mode heatmap / lines | $\mathbf{V}_{nk} = (\mathbf{V})_{nk}$ from SVD | Modal concentration: $\sum_k V_{nk}^2 = 1.0$ | Projection purity: $|V_{nn}| \approx 1.0$ | Incoherent mixing coefficients, rank collapse |
| **Figure 12** | `temporal_overlap` | $N \times N$ matrix heatmap | $\mathbf{O}^V = \mathbf{V}^T \mathbf{V} = \mathbf{I}_N$ | SVD isometry: $\|\mathbf{V}^T \mathbf{V} - \mathbf{I}_N\|_F < 10^{-12}$ | N/A (mathematical SVD integrity) | Loss of numerical unitarity in singular value decomposition |
| **Figure 13** | `cross_temporal` | $N \times N$ absolute projection heatmap | $P_{nk} = |\langle \mathbf{e}_n, \mathbf{v}_k \rangle| = |V_{nk}|$ | Spatial-temporal consistency | Unitary mapping fidelity | Dispersion of snapshot energy across non-principal singular modes |
| **Figure 14** | `quantum_state_fidelity` [Salmanogli 2026] | 1D profile + multi-state radar chart (`magma` / `plasma`) | $F_n = \|\langle \hat{\psi}_n^\theta, \psi_n^{\text{true}} \rangle_W\|^2$, $D_n = \frac{1}{2}\int \|\rho_n^\theta(x) - \rho_n^{\text{true}}(x)\| dx$ | Pure state normalization: $F_n \to 1.0$, $D_n \to 0.0$ | State fidelity $F_n > 0.995$, Trace distance $D_n < 0.02$ | Gauge phase drift, tail probability density distortion |
| **Figure 15** | `hamiltonian_reconstruction` [Salmanogli 2026] | 2D matrix error heatmap (`RdBu_r`) | $\Delta \mathbf{H} = |\mathbf{H}_\theta - \mathbf{H}_{\text{true}}|$, $\mathbf{H}_\theta = -\frac{1}{2}\mathbf{D}_{xx} + \text{diag}(V_\theta)$ | Galerkin decoupling: $\|\mathbf{H}_{\text{pod}} - \text{diag}(\mathbf{H}_{\text{pod}})\|_F / \|\mathbf{H}_{\text{pod}}\|_F < 10^{-3}$ | Relative Hamiltonian error: $\|\mathbf{H}_\theta - \mathbf{H}_{\text{true}}\|_F / \|\mathbf{H}_{\text{true}}\|_F < 0.01$ | Non-Hermitian asymmetry, kinetic-potential energy imbalance |
| **Figure 16** | `excitation_stability` [Salmanogli 2026] | Trajectory loss & steady-state perturbation spectrum | $\mathcal{L}_{\text{steady}}(\delta, \omega_{\text{chirp}}) = \|\exp(-i\mathbf{H}_\theta t)\psi_0 - \exp(-i\mathbf{H}_{\text{true}} t)\psi_0\|^2$ | Stability margin: $\max_t \|\psi(t)\|^2 = 1.0 \pm 10^{-5}$ | Trajectory divergence rate $< 10^{-3}$ | Steady-state loss plateau under randomized initial states and chirp excitations |

---

## Part 2: Additional Success Metrics & Physics/Math Bridge Diagnostics

The following eight metrics and diagnostic plots expose critical hidden failure modes (such as gauge drifts, operator inconsistency, boundary under-determinism, and Hamiltonian parameter bias) and bridge physical, geometric, and quantum neural principles.

```text
+--------------------------------------------------------------------------------------------------------------------------+
|                                           PROPOSED ADVANCED SUCCESS METRICS                                              |
+--------------------------------------------------------------------------------------------------------------------------+
| 1. Hamiltonian Commutator Invariance Metric:   M_comm = || [H_theta, P_n] ||_F / (||H_theta||_F ||P_n||_F)               |
| 2. Galerkin Energy Leakage Index:              E_off  = || H_pod - diag(H_pod) ||_F / || H_pod ||_F                      |
| 3. Local Fisher-Physics Density Weighting:     I_FP(x) = sum_n |psi_n(x)|^2 * |R_n(x)|^2                                  |
| 4. Quantum Virial Theorem Consistency Ratio:   eta_virial = 2<T>_n / <x dV/dx>_n - 1                                     |
| 5. Fubini-Study Quantum Metric Tensor:         g_mu_nu = Re<d_mu psi | d_nu psi> - <d_mu psi|psi><psi|d_nu psi>          |
| 6. Wigner-Weyl Phase-Space Bivector Metric:    W_psi(x,p) = (1/pi hbar) int psi^*(x+y) psi(x-y) e^(2i py/hbar) dy        |
| 7. Uhlmann-Jozsa Fidelity & Trace Distance:   F_n = |<psi_n^theta, psi_n^true>|^2,  D_n = 1/2 int |rho_theta - rho_true| |
| 8. Hamiltonian Learning Coefficient Error:     eps_H = || H_theta - H_true ||_F / || H_true ||_F                         |
+--------------------------------------------------------------------------------------------------------------------------+
```

### 1. Hamiltonian Commutator Residual & Density Invariance ($\mathcal{M}_{\text{comm}}$)
- **Plot/Colormap:** 2D Heatmap of the matrix commutator $[\mathbf{H}_\theta, \mathbf{P}_n] = \mathbf{H}_\theta \mathbf{P}_n - \mathbf{P}_n \mathbf{H}_\theta$, where $\mathbf{P}_n = \hat{\boldsymbol{\psi}}_n (\hat{\boldsymbol{\psi}}_n)^T \mathbf{W}$.
- **Physical Concept:** In quantum mechanics, stationary eigenstates are generators of constant density under unitary time evolution. By Heisenberg's equation of motion, $[\hat{H}, \hat{\rho}_n] = 0$.
- **Mathematical Derivation:**
  $$\mathcal{M}_{\text{comm}}^{(n)} = \frac{\|\mathbf{H}_\theta \mathbf{P}_n - \mathbf{P}_n \mathbf{H}_\theta\|_F}{\|\mathbf{H}_\theta\|_F \|\mathbf{P}_n\|_F} \, , \qquad \mathbf{P}_n = \hat{\boldsymbol{\psi}}_n^\theta (\hat{\boldsymbol{\psi}}_n^\theta)^T \mathbf{W}$$
- **Failure Mode Exposed:** Exposes operator-state incompatibility. A network may superficially match $\hat{\psi}_n^\theta$ and $E_n^\theta$, but if $\mathcal{M}_{\text{comm}}^{(n)} > 10^{-2}$, the learned potential does not form a true dynamical invariant generator for the state.

### 2. Galerkin Off-Diagonal Decoupling & Energy Leakage Index ($\mathcal{E}_{\text{off}}$)
- **Plot/Colormap:** 2D Heatmap of $\mathbf{H}_{\text{pod}} = (\mathbf{U}^{\text{phys}})^T \mathbf{W} \mathbf{H}_\theta \mathbf{U}^{\text{phys}}$.
- **Physical Concept:** Hamiltonian diagonalizability in the POD empirical eigenbasis. If POD spatial modes span the true physical eigenspaces, the modal Hamiltonian must be strictly diagonal.
- **Mathematical Derivation:**
  $$\mathcal{E}_{\text{off}} = \frac{\|\mathbf{H}_{\text{pod}} - \text{diag}(\mathbf{H}_{\text{pod}})\|_F}{\|\mathbf{H}_{\text{pod}}\|_F}$$
- **Failure Mode Exposed:** Inter-modal cross-talk and Hamiltonian leakage. Measures exact physical energy dissipation between modes when using the POD basis for reduced-order modeling.

### 3. Local Fisher-Physics Density Weighting ($I_{\text{FP}}(x)$)
- **Plot/Colormap:** 1D dual-axis plot comparing learned density envelope $\rho(x) = \sum_n |\hat{\psi}_n^\theta(x)|^2$ against local residual intensity $|R_n(x)|^2$.
- **Physical Concept:** Solves the "Shadow Illusion" of inverse physics problems: the neural potential $V_\theta(x)$ is unconstrained in regions where $\rho(x) \approx 0$.
- **Mathematical Derivation:**
  $$I_{\text{FP}}(x) = \sum_{n=0}^{N-1} |\hat{\psi}_n^\theta(x)|^2 \cdot |R_n(x)|^2 \, , \qquad \mathcal{S}_{\text{obs}} = \int_{-L/2}^{L/2} I_{\text{FP}}(x) \, dx$$
- **Failure Mode Exposed:** Boundary hallucination. Identifies whether low residual values in the outer domain are genuine physical solutions or trivial $\psi(x) \to 0$ numerical artifacts.

### 4. Quantum Virial Theorem Consistency Ratio ($\eta_{\text{virial}}$)
- **Plot/Colormap:** Bar chart of $\eta_{\text{virial}}^{(n)}$ per state $n$.
- **Physical Concept:** Fundamental quantum virial theorem for stationary states in a power-law potential $V(x) \propto x^k$ (for harmonic oscillator $k=2$, $2\langle T \rangle = 2\langle V \rangle = \langle x V' \rangle$).
- **Mathematical Derivation:**
  $$\langle T \rangle_n = \frac{1}{2} \int \left|\frac{d\hat{\psi}_n^\theta}{dx}\right|^2 dx \, , \qquad \langle x V'\rangle_n = \int x \frac{dV_\theta}{dx} |\hat{\psi}_n^\theta|^2 dx$$
  $$\eta_{\text{virial}}^{(n)} = \left| \frac{2\langle T \rangle_n}{\langle x V'\rangle_n} - 1 \right|$$
- **Failure Mode Exposed:** Global non-physical force balance. Evaluates whether $V_\theta(x)$ is physically genuine without requiring access to the analytical ground-truth potential.

### 5. Fubini-Study Quantum Metric Tensor on Parameter Manifold ($g_{\mu\nu}$)
- **Plot/Colormap:** 2D heatmap or spectrum of the parameter metric $g_{\mu\nu}(\boldsymbol{\theta})$.
- **Physical Concept:** Measures the Riemannian distance between quantum state outputs across parameter perturbations, establishing the Information Geometry / Quantum Natural Gradient framework.
- **Mathematical Derivation:**
  $$g_{\mu\nu}(\boldsymbol{\theta}) = \text{Re}\langle \partial_\mu \psi(\boldsymbol{\theta}) | \partial_\nu \psi(\boldsymbol{\theta}) \rangle - \langle \partial_\mu \psi(\boldsymbol{\theta}) | \psi(\boldsymbol{\theta}) \rangle \langle \psi(\boldsymbol{\theta}) | \partial_\nu \psi(\boldsymbol{\theta}) \rangle$$
  $$\kappa(g) = \frac{\lambda_{\max}(g)}{\lambda_{\min}(g)}$$
- **Failure Mode Exposed:** Over-parameterization, gauge redundancy, and flat optimization valleys (sloppy parameter directions).

### 6. Phase-Space Wigner-Weyl Distribution & Geometric Exterior Bivector Volume
- **Plot/Colormap:** 2D Phase-space contour plot $W_{\psi_n}(x, p)$ in $(x, p)$ plane (`RdBu_r` diverging colormap).
- **Physical & Geometric Concept:** Wigner quasiprobability distribution in phase space and its symplectic area element in Geometric Algebra $B = dx \wedge dp$ ($i \leftrightarrow e_1 e_2$).
- **Mathematical Derivation:**
  $$W_{\hat{\psi}_n^\theta}(x, p) = \frac{1}{\pi \hbar} \int_{-\infty}^{\infty} \hat{\psi}_n^\theta(x + y)^* \hat{\psi}_n^\theta(x - y) e^{2i p y / \hbar} dy$$
  $$\delta_{\text{Wigner}}^{(n)} = \iint_{W < 0} |W_{\hat{\psi}_n^\theta}(x, p)| \, dx \, dp$$
- **Failure Mode Exposed:** High-frequency unphysical ripples in tails and phase-space distortion. A genuine harmonic state $\psi_n$ has exactly $n$ negative interference rings.

### 7. Uhlmann-Jozsa Quantum State Fidelity & Trace Distance Discrepancy [Salmanogli 2026]
- **Plot/Colormap:** Multi-state radar plot and 1D local trace density error $| \rho_n^\theta(x) - \rho_n^{\text{true}}(x) |$.
- **Physical Concept:** Information-theoretic fidelity between learned quantum state ansatz and ground-truth eigenstates as formulated in quantum tomography and Hamiltonian learning (Salmanogli 2026).
- **Mathematical Derivation:**
  For pure stationary eigenstates:
  $$\mathcal{F}_n(\hat{\psi}_n^\theta, \psi_n^{\text{true}}) = \left| \int \hat{\psi}_n^\theta(x)^* \psi_n^{\text{true}}(x) \, dx \right|^2 = |\langle \hat{\psi}_n^\theta, \psi_n^{\text{true}} \rangle_W|^2$$
  The trace distance on the diagonal probability projection:
  $$\mathcal{D}_n(\rho_n^\theta, \rho_n^{\text{true}}) = \frac{1}{2} \int_{-\infty}^{\infty} \left| \rho_n^\theta(x) - \rho_n^{\text{true}}(x) \right| dx = \frac{1}{2} \|\boldsymbol{\rho}_n^\theta - \boldsymbol{\rho}_n^{\text{true}}\|_{1, W}$$
- **Failure Mode Exposed:** Gauge phase ambiguity vs. true structural state discrepancy; separates global $U(1)$ phase factor rotation from amplitude deformation.

### 8. Hamiltonian Learning Coefficient Error & Perturbation Stability [Salmanogli 2026]
- **Plot/Colormap:** 2D absolute difference matrix $|\mathbf{H}_\theta - \mathbf{H}_{\text{true}}|$ and chirped excitation loss trajectory $\mathcal{L}_{\text{dyn}}(t)$.
- **Physical Concept:** Evaluating the accuracy of the reconstructed Hamiltonian operator under full dynamical evolution trajectories and chirped/randomized initial quantum states (Salmanogli 2026).
- **Mathematical Derivation:**
  $$\varepsilon_H = \frac{\|\mathbf{H}_\theta - \mathbf{H}_{\text{true}}\|_F}{\|\mathbf{H}_{\text{true}}\|_F} \, , \qquad \mathbf{H}_\theta = -\frac{1}{2}\mathbf{D}_{xx} + \text{diag}(V_\theta)$$
  Under dynamic state propagation with randomized excitation:
  $$\mathcal{L}_{\text{traj}}(T) = \frac{1}{T}\int_0^T \mathcal{D}\left( e^{-i\mathbf{H}_\theta t} \psi_0, \, e^{-i\mathbf{H}_{\text{true}} t} \psi_0 \right) dt$$
- **Failure Mode Exposed:** Steady-state loss floors and optimization stagnancy under combined training perturbations.

---

## Part 3: POD Analysis & Post-Training Derivation Worksheet (Linear Algebra & Geometric Algebra Review)

```text
==========================================================================================================================
                                       DERIVATION WORKSHEET: DISCRETE WEIGHTED POD
==========================================================================================================================

[Step 1: Continuous Hilbert Space]        L^2(R) Inner Product:  <f, g> = \int f^*(x) g(x) dx
                                                               |
[Step 2: Discrete Quadrature Metric]      Matrix W = diag(w_0 dx, w_1 dx, ..., w_{M-1} dx)
                                          <f, g>_W = f^T W g = (W^(1/2) f)^T (W^(1/2) g)
                                                               |
[Step 3: Weighted Snapshot Matrix]        Psi^\theta = [psi_0, psi_1, psi_2] \in R^{M x N}
                                          Psi_w = W^(1/2) Psi^\theta
                                                               |
[Step 4: Thin SVD in Isometry Space]      Psi_w = U Sigma V^T,   U \in R^{M x N}, Sigma \in R^{N x N}, V \in R^{N x N}
                                          U^T U = I_N,   V^T V = I_N,   V V^T = I_N
                                                               |
[Step 5: Physical Mode Recovery]          u_k^phys = W^(-1/2) u_k   ==>  <u_i^phys, u_j^phys>_W = u_i^T W u_j = \delta_{ij}
                                                               |
[Step 6: Hamiltonian Galerkin Proj.]      H_pod = (U^phys)^T W H_\theta U^phys = U^T W^(1/2) H_\theta W^(-1/2) U
                                                               |
[Step 7: QNN Emulation & Fidelity]        F_n = |(U^phys_n)^T W psi_n^true|^2,  ||H_pod - diag(E_n)||_F -> 0
==========================================================================================================================
```

### 1. Quadrature-Weighted Inner Product and Metric Space Discretization
Let $\Omega = [-L/2, L/2]$ be discretized into $M$ nodes $x_0, x_1, \dots, x_{M-1}$ with spacing $\Delta x = \frac{L}{M-1}$. The trapezoidal quadrature approximation to $\langle f, g \rangle_{L^2} = \int_{-L/2}^{L/2} f^*(x) g(x) dx$ is:
$$\langle f, g \rangle_W = \mathbf{f}^T \mathbf{W} \mathbf{g} = \sum_{m=0}^{M-1} w_m \Delta x \, f(x_m) g(x_m)$$
where $\mathbf{W} \in \mathbb{R}^{M \times M}$ is the positive-definite diagonal metric matrix:
$$\mathbf{W} = \text{diag}\left( \frac{1}{2}\Delta x, \, \Delta x, \, \Delta x, \, \dots, \, \Delta x, \, \frac{1}{2}\Delta x \right)$$
Because $\mathbf{W}$ is diagonal and strictly positive, its symmetric square root and inverse square root are:
$$\mathbf{W}^{1/2} = \text{diag}\left( \sqrt{w_m \Delta x} \right) \, , \qquad \mathbf{W}^{-1/2} = \text{diag}\left( \frac{1}{\sqrt{w_m \Delta x}} \right)$$

### 2. Weighted Snapshot SVD and Isometry Mapping
Given the neural snapshot matrix $\mathbf{\Psi}^\theta \in \mathbb{R}^{M \times N}$ ($N = N_{\text{states}}$):
$$\mathbf{\Psi}^\theta = \begin{bmatrix} \hat{\boldsymbol{\psi}}_0^\theta & \hat{\boldsymbol{\psi}}_1^\theta & \dots & \hat{\boldsymbol{\psi}}_{N-1}^\theta \end{bmatrix}$$

- **Transform to Standard Euclidean Metric Space:**
  $$\mathbf{\Psi}_w = \mathbf{W}^{1/2} \mathbf{\Psi}^\theta \in \mathbb{R}^{M \times N}$$

- **Standard Thin Singular Value Decomposition:**
  $$\mathbf{\Psi}_w = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
  where:
  - $\mathbf{U} \in \mathbb{R}^{M \times N}$, with $\mathbf{U}^T \mathbf{U} = \mathbf{I}_N$ (Euclidean orthonormal columns).
  - $\mathbf{\Sigma} = \text{diag}(\sigma_0, \sigma_1, \dots, \sigma_{N-1}) \in \mathbb{R}^{N \times N}$, with $\sigma_0 \ge \sigma_1 \ge \dots \ge \sigma_{N-1} \ge 0$.
  - $\mathbf{V} \in \mathbb{R}^{N \times N}$, with $\mathbf{V}^T \mathbf{V} = \mathbf{V} \mathbf{V}^T = \mathbf{I}_N$.

- **Physical Spatial Basis Mode Recovery:** Transform back to the physical $L^2$ functional space:
  $$\mathbf{U}^{\text{phys}} = \mathbf{W}^{-1/2} \mathbf{U} \in \mathbb{R}^{M \times N} \implies \mathbf{u}_k^{\text{phys}} = \mathbf{W}^{-1/2} \mathbf{u}_k$$

- **Proof of Physical $L^2$ Orthonormality:**
  $$(\mathbf{U}^{\text{phys}})^T \mathbf{W} \mathbf{U}^{\text{phys}} = (\mathbf{W}^{-1/2} \mathbf{U})^T \mathbf{W} (\mathbf{W}^{-1/2} \mathbf{U}) = \mathbf{U}^T (\mathbf{W}^{-1/2} \mathbf{W} \mathbf{W}^{-1/2}) \mathbf{U} = \mathbf{U}^T \mathbf{I}_M \mathbf{U} = \mathbf{U}^T \mathbf{U} = \mathbf{I}_N$$

### 3. Modal Hamiltonian Construction via Galerkin Projection
- **Discrete Kinetic Operator:** 3-point finite-difference central stencil for $\mathbf{D}_{xx} \approx \frac{d^2}{dx^2}$:
  $$\mathbf{T} = -\frac{1}{2} \mathbf{D}_{xx} = -\frac{1}{2 \Delta x^2} \begin{bmatrix} -2 & 1 & 0 & \dots & 0 \\ 1 & -2 & 1 & \dots & 0 \\ \vdots & \ddots & \ddots & \ddots & \vdots \\ 0 & \dots & 1 & -2 & 1 \\ 0 & \dots & 0 & 1 & -2 \end{bmatrix}$$
- **Discrete Physical Hamiltonian:**
  $$\mathbf{H}_\theta = \mathbf{T} + \text{diag}(V_\theta(x_0), V_\theta(x_1), \dots, V_\theta(x_{M-1}))$$
- **Galerkin Projection:**
  $$\mathbf{H}_{\text{pod}} = (\mathbf{U}^{\text{phys}})^T \mathbf{W} \mathbf{H}_\theta \mathbf{U}^{\text{phys}} = \mathbf{U}^T \mathbf{W}^{1/2} \mathbf{H}_\theta \mathbf{W}^{-1/2} \mathbf{U} \in \mathbb{R}^{N \times N}$$
  - **Diagonal Entries:** Modal energy expectations $(H_{\text{pod}})_{kk} = \langle u_k^{\text{phys}}, \hat{H}_\theta u_k^{\text{phys}} \rangle_W \approx E_k$.
  - **Off-Diagonal Entries:** Cross-coupling amplitudes $(H_{\text{pod}})_{kj} = \langle u_k^{\text{phys}}, \hat{H}_\theta u_j^{\text{phys}} \rangle_W \to 0$.

### 4. Quantum Density Matrix Dynamics & Hamiltonian Learning Bridge [Salmanogli 2026]
In the framework of Salmanogli (2026), Hamiltonian learning reconstructs the generator of unitary dynamics:
$$\rho(t) = e^{-i H t} \rho(0) e^{i H t}$$
In the reduced POD subspace spanned by $\{\mathbf{u}_k^{\text{phys}}\}_{k=0}^{N-1}$, the spatial state vector $\mathbf{c}(t) \in \mathbb{C}^N$ evolves according to the low-dimensional Schrödinger equation:
$$i \frac{d\mathbf{c}(t)}{dt} = \mathbf{H}_{\text{pod}} \mathbf{c}(t)$$
Because $\mathbf{H}_{\text{pod}} \approx \text{diag}(E_0, E_1, \dots, E_{N-1})$, the trajectory density matches the exact physical evolution without dimensional curse. Evaluating the state fidelity $\mathcal{F}(t) = |\mathbf{c}(t)^\dagger \mathbf{c}_{\text{true}}(t)|^2$ confirms both static eigenvalue alignment and dynamic unitary fidelity.

### 5. Geometric Algebra ($\mathcal{Cl}_{3,0}^+$) Review & Bridging
- **Geometric Product:** $ab = a \cdot b + a \wedge b$ (scalar symmetric contraction + bivector antisymmetric wedge product).
- **Quantum State as Even Multivector:** $\psi(x) = \sqrt{\rho(x)} \exp(-I \phi(x)) \in \mathcal{Cl}_{3,0}^+$, where $I = e_1 e_2 e_3$ ($I^2 = -1$).
- **Exterior Outer Product ($\wedge$) & Basis Orthogonality Volume:**
  $$\|\mathbf{u}_0 \wedge \mathbf{u}_1 \wedge \dots \wedge \mathbf{u}_{N-1}\| = \sqrt{\det(\mathbf{S})} = \prod_{k=0}^{N-1} \sigma_k$$
  If mode mixing or rank collapse occurs, $\det(\mathbf{S}) \to 0$, signaling that the $N$-volume has collapsed.

---

## Part 4: Summary Table of Post-Training Verification Protocol

| Verification Stage | Tool / Operator | Target Value | Tolerable Threshold | Red Flag / Action |
| :--- | :--- | :--- | :--- | :--- |
| **1. Normalization** | $\int \rho_n^\theta(x) dx$ | $1.0000$ | $|1.0 - \int \rho dx| < 10^{-3}$ | Check trapezoidal quadrature weights $\mathbf{W}$ |
| **2. Orthogonality** | $\langle \hat{\psi}_i^\theta, \hat{\psi}_j^\theta \rangle_W$ | $\delta_{ij}$ | $\|\mathbf{S} - \mathbf{I}\|_F < 10^{-4}$ | Increase Gram-Schmidt projection in forward pass |
| **3. Energy Ordering** | $E_n^\theta - E_{n-1}^\theta$ | $\hbar \omega = 1.0$ (HO) | $E_n - E_{n-1} > 0.0$ | Re-weight $\lambda_{\text{order}}$ penalty |
| **4. POD Dimension** | $\sigma_k / \sigma_0$ | $< 10^{-3}$ ($k \ge 3$) | $\sigma_2 / \sigma_0 > 10^{-2}$ | Investigate noise floor; tune learning rate |
| **5. Galerkin Decoupling** | $\mathcal{E}_{\text{off}} = \|\mathbf{H}_{\text{pod}} - \text{diag}\|_F / \|\mathbf{H}_{\text{pod}}\|_F$ | $< 10^{-4}$ | $\mathcal{E}_{\text{off}} < 10^{-2}$ | State-operator mismatch; enforce higher $\lambda_{\text{phys}}$ |
| **6. Virial Balance** | $\eta_{\text{virial}} = |2\langle T \rangle / \langle x V' \rangle - 1|$ | $0.000$ | $\eta_{\text{virial}} < 0.05$ | Potential curvature inaccurate in data-sparse boundary |
| **7. External Alignment** | $|\langle u_k^{\text{phys}}, \psi_k^{\text{true}} \rangle_W|$ | $1.000$ | $|\langle u_k, \psi_k \rangle| > 0.990$ | Retrain with adjusted initialization seed |
| **8. Quantum State Fidelity** [Salmanogli 2026] | $\mathcal{F}_n = |\langle \hat{\psi}_n^\theta, \psi_n^{\text{true}} \rangle_W|^2$ | $1.000$ | $\mathcal{F}_n \ge 0.995$ | Mode deformation; check potential depth |
| **9. Trace Distance Discrepancy** [Salmanogli 2026] | $\mathcal{D}_n = \frac{1}{2}\int |\rho_n^\theta - \rho_n^{\text{true}}| dx$ | $0.000$ | $\mathcal{D}_n \le 0.020$ | Unphysical density tail oscillation |
| **10. Hamiltonian Error** [Salmanogli 2026] | $\varepsilon_H = \|\mathbf{H}_\theta - \mathbf{H}_{\text{true}}\|_F / \|\mathbf{H}_{\text{true}}\|_F$ | $< 0.005$ | $\varepsilon_H \le 0.020$ | Discretization error or boundary drift |

---

## Part 5: Action Plan for Final Major Release (Figures, CLI, and Demo Notebook Blocks)

This section lays out the implementation roadmap for integrating the new diagnostics, metrics from Salmanogli (2026), CLI extensions, and interactive research-notebook blocks for the Zensical documentation website.

### Phase 1: Diagnostic Visualizations Module Enhancement (`src/visualizations.py`)
- [ ] **Figure 14 (`plot_quantum_fidelity_trace_distance`)**:
  - Implement 2-panel figure: (Left) Bar chart of pure state fidelity $\mathcal{F}_n$ and trace distance $\mathcal{D}_n$ per state $n$; (Right) 1D spatial density absolute error curves $|\rho_n^\theta(x) - \rho_n^{\text{true}}(x)|$ with shaded confidence bands.
  - Colormap: `magma` / `plasma`.
  - Artifact path: `artifacts/runs/<run_id>/figures/quantum_state_fidelity.png`.
- [ ] **Figure 15 (`plot_hamiltonian_reconstruction_matrix`)**:
  - Implement 3-panel figure: (a) Ground truth discrete Hamiltonian $\mathbf{H}_{\text{true}}$, (b) Learned neural Hamiltonian $\mathbf{H}_\theta$, (c) Absolute difference matrix $|\mathbf{H}_\theta - \mathbf{H}_{\text{true}}|$ and Galerkin projected $\mathbf{H}_{\text{pod}}$.
  - Colormap: `RdBu_r` (diverging) and `viridis` (positive).
  - Artifact path: `artifacts/runs/<run_id>/figures/hamiltonian_reconstruction.png`.
- [ ] **Figure 16 (`plot_excitation_response_perturbation`)**:
  - Implement trajectory evolution plot comparing time-dependent probability density propagation under $\mathbf{H}_\theta$ vs. $\mathbf{H}_{\text{true}}$ for randomized superpositions and chirped excitation wavepackets.
  - Artifact path: `artifacts/runs/<run_id>/figures/excitation_stability.png`.
- [ ] **Update Master Plot Generator (`plot_all_diagnostics`)**:
  - Ensure all 16 figures are saved uniformly when `--show-figures` or `--figures-dir` is passed to the CLI.

### Phase 2: CLI Interface & Metric Extraction Upgrades (`cli/cli_train.py` & `cli/extract_metrics.py`)
- [ ] **CLI Training Flags (`cli/cli_train.py`)**:
  - Add `--eval-fidelity`: Computes $\mathcal{F}_n$ and $\mathcal{D}_n$ at validation checkpoints.
  - Add `--eval-hamiltonian`: Generates $\mathbf{H}_{\text{pod}}$, $\mathcal{E}_{\text{off}}$, and $\varepsilon_H$ reports.
  - Add `--save-qnn-benchmarks`: Exports JSON summary formatted for Hamiltonian learning benchmarks.
- [ ] **Metric Extraction Tool (`cli/extract_metrics.py`)**:
  - Extend `training_analysis.csv` to export columns:
    - `fidelity_state_0`, `fidelity_state_1`, `fidelity_state_2`
    - `trace_distance_state_0`, `trace_distance_state_1`, `trace_distance_state_2`
    - `hamiltonian_rel_frobenius_error`, `galerkin_leakage_index`
    - `virial_ratio_state_0`, `virial_ratio_state_1`, `virial_ratio_state_2`
  - Extend `pod_metrics.csv` to log modal Hamiltonian eigenvalues and Frobenius condition numbers.
- [ ] **Database Logger Updates (`src/db_logger.py`)**:
  - Add SQLite columns in `data/training_runs.db` for `mean_fidelity`, `max_trace_distance`, and `hamiltonian_error`.

### Phase 3: Interactive Demo Notebook Blocks (`notebooks/load_run_analysis.ipynb` & `demo.ipynb`)
- [ ] **Block 5: Quantum State Fidelity & Information Geometry**:
  - Interactive cell computing $\mathcal{F}_n = |\langle\hat{\psi}_n^\theta, \psi_n^{\text{true}}\rangle_W|^2$ and $\mathcal{D}_n = \frac{1}{2}\int |\rho_n^\theta - \rho_n^{\text{true}}| dx$.
  - Generates Figure 14 and displays tabular comparison.
- [ ] **Block 6: Discrete Hamiltonian Reconstruction & Galerkin Decoupling**:
  - Computes $\mathbf{H}_\theta = -\frac{1}{2}\mathbf{D}_{xx} + \text{diag}(V_\theta)$ and $\mathbf{H}_{\text{pod}} = (\mathbf{U}^{\text{phys}})^T \mathbf{W} \mathbf{H}_\theta \mathbf{U}^{\text{phys}}$.
  - Renders Figure 15 error heatmaps with colorbars and off-diagonal leakage metrics.
- [ ] **Block 7: Unitary Trajectory Emulation & Perturbation Benchmark (Salmanogli 2026)**:
  - Propagates initial wavepacket $\psi_0(x)$ under matrix exponential $\mathbf{U}(t) = \exp(-i \mathbf{H}_\theta t)$.
  - Verifies stability against steady-state loss floors and randomized initializations.
  - Renders Figure 16 dynamic trajectory comparison.

### Phase 4: Zensical Research-Notebook Website Documentation Integration
- [ ] **Static Site Generation Sync**:
  - Export rendered notebook cells from `notebooks/load_run_analysis.ipynb` with interactive Plotly/Matplotlib visual figures into the Zensical markdown site docs (`docs/` directory).
- [ ] **Mathematical Theory Cards**:
  - Format LaTeX cards for Part 2 & Part 3 (Geometric Algebra multivector representation, SVD isometry mapping, and Salmanogli Hamiltonian learning framework).
- [ ] **Bibliography & Reference Sync**:
  - Ensure `references.bib` key `@article{salmanogli2026QNN, ...}` is cross-referenced in website footer and publication notes.
