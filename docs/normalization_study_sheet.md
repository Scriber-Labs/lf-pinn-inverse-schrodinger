# 🧠 Study Sheet: Normalization & Geometric Constraints
**Project:** Inverse Schrödinger PINN (PINN-Inverse-Schrodinger)

---

## 🧭 The Golden Rule
> **"Normalize the Domain (Inputs), Constrain the Range (Outputs)."**
*   **Domain:** The coordinate system ($x$). Must be stable and predictable.
*   **Range:** The model predictions ($\psi, V, E$). Must obey physical laws (Probability=1, Orthogonality=0) *before* the loss is calculated.

---

## 1. Input Grid Normalization (`Domain Normalization`)
*   **Technical Step:** Use `make_grid` to keep $x$ within a reasonable range (e.g., $[-5, 5]$).
*   **Why it matters:** Neural networks are essentially high-dimensional curve fitters. If $x$ ranges from $0$ to $10,000$, the gradients become "stiff" and training fails. A centered, scaled domain ensures smooth optimization.
*   **Physics Link:** Finite-difference stencils (for $\psi''$) require a fixed, uniform $dx$ to be accurate.

## 2. L2 Normalization by Construction (`Range Constraint`)
*   **Technical Step:** Wrapping MLPs in `NormalizedWavefunctionNet`.
*   **Why it matters:** In QM, $\int |\psi|^2 dx = 1$ is a law, not a suggestion. By building this into the model, we prevent the "Trivial Solution" where the model minimizes the Schrödinger residual by simply setting $\psi = 0$ everywhere.
*   **Key Concept:** **Hard Constraint**. The optimizer *cannot* break this rule.

## 3. Gram-Schmidt Orthonormalization (`Range Constraint`)
*   **Technical Step:** `model.psi_theta` uses sequential subtraction of projections.
*   **Why it matters:** Prevents **Mode Collapse**. Without this, all $N$ networks in your `ModuleList` would likely learn the same ground-state (lowest energy) solution. This forces them to be unique and non-overlapping.
*   **Key Math:** $\langle \psi_i, \psi_j \rangle = \delta_{ij}$.

## 4. Numerical Safeguard ($\epsilon = 10^{-8}$)
*   **Technical Step:** Adding $\epsilon$ inside the square root: $1 / \sqrt{\text{Norm} + \epsilon}$.
*   **Why it matters:** At the very first "tick" of training, weights are random. If a network outputs all zeros, the norm is zero. Dividing by zero gives `NaN`, which "infects" the entire model and kills the run. $\epsilon$ is your insurance policy.

## 5. Phase Alignment (`Post-processing`)
*   **Technical Step:** Checking if `l2_inner_product(psi_learned, psi_true, dx) < 0` and flipping the sign for consistency.
*   **Why it matters:** The Schrödinger equation doesn't care about $\pm$ signs (it only cares about $|\psi|^2$). However, humans (and error metrics) do. We align signs *only for visualization* using `l2_inner_product(..., dx)` so we can see if the shapes match consistently.
*   **Critical Note:** Never do this *inside* the training loop; it confuses the optimizer.

## 6. POD & Physical Weighting (`Analysis Alignment`)
*   **Technical Step:** Scaling SVD modes by $1/\sqrt{w \cdot dx}$ (or pre-weighting snapshots by $\sqrt{w \cdot dx}$ with trapezoidal endpoint weights $w$).
*   **Why it matters:** SVD is "unitless" linear algebra. Wavefunctions are "physical" functions. To compare an abstract SVD vector to a physical wavefunction, you must account for the integration measure with quadrature weights ($w \cdot dx$).
*   **The Trap:** If you skip this, your POD modes will look like the right shape but will be on the wrong scale ($1.0$ vs $1/\sqrt{w \cdot dx}$), leading to incorrect error reports.

---

### 🚨 Quick-Check Table for Code Reviews

| If you see... | Ask yourself... | Correct behavior |
| :--- | :--- | :--- |
| `psi = model(x)` | Is it normalized? | Should be wrapped in `NormalizedWavefunctionNet`. |
| `loss = sum(psi**2)` | Is the measure included? | Use `l2_inner_product(psi, psi, dx)` (Trapezoidal). |
| `U, S, V = svd(A)` | Is this physical POD? | Must weight by `sqrt(w * dx)` first (Trapezoidal) or scale $U$ by $1/\sqrt{w \cdot dx}$ after. |
| `NaN` in loss | Is there an epsilon? | Check all divisions and square roots in `utils.py`. |

---

### 🎯 Goal
By keeping these in mind, you ensure that the **Hypothesis Space** of your neural network is restricted only to **Physically Plausible** solutions. The optimizer's only job then is to find the *best* physical solution, not to learn what "physical" means.
