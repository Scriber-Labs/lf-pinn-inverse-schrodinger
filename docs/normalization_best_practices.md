# Normalization Best Practices for Inverse Schrödinger PINNs

## 1. Core Philosophy: Construction vs. Constraint
In Physics-Informed Neural Networks (PINNs), there are two primary ways to handle the normalization ($\int |\psi|^2 dx = 1$) and orthogonality ($\int \psi_i^* \psi_j dx = \delta_{ij}$) requirements.

### A. Normalization by Construction (Hard Constraints)
Instead of asking the loss function to "find" normalized solutions, we wrap the network output in a normalization layer.
- **Single Mode:** $\psi_{norm}(x) = \frac{\psi_{raw}(x)}{\sqrt{\int |\psi_{raw}|^2 dx}}$
- **Multiple Modes (Orthonormalization):** Use a sequential Gram-Schmidt process within the `forward` pass.
  1. Normalize $\psi_1$.
  2. Project $\psi_1$ out of $\psi_2$, then normalize the remainder.
  3. Repeat for $\psi_n$.
- **Advantage:** Guarantees physical validity at every iteration.
- **Disadvantage:** Can lead to vanishing gradients if the raw network output becomes very small or highly redundant.

### B. Normalization by Penalty (Soft Constraints)
Add terms to the loss function: $\mathcal{L}_{norm} = \sum_i (\int |\psi_i|^2 dx - 1)^2$.
- **Advantage:** Gentler optimization landscape.
- **Disadvantage:** Never perfectly satisfied; requires careful hyperparameter tuning ($\lambda_{norm}$).

---

## 2. Proper Orthogonal Decomposition (POD) as a Diagnostic
POD (effectively SVD for physical fields) is the gold standard for evaluating how well a PINN is learning the underlying Hilbert space.

### The "Physical SVD" Requirement
A standard SVD assumes a Euclidean metric ($I$). Physical wavefunctions live in $L^2$, where the inner product involves a spatial measure ($dx$ or quadrature weights $w$).
To perform a **Physical POD**:
1. **Weighting:** Pre-multiply the snapshot matrix $\Psi$ by the square root of the quadrature weights: $\Psi_w = \text{diag}(\sqrt{w \cdot dx}) \Psi$.
2. **SVD:** Perform standard SVD on $\Psi_w = U_w S V^T$.
3. **Un-weighting:** Recover physical modes: $U_{phys} = \text{diag}(1/\sqrt{w \cdot dx}) U_w$.
This ensures that the resulting modes $U_{phys}$ are orthonormal with respect to the physical $L^2$ inner product, not just the vector dot product.

### Key POD Metrics
- **Singular Value Decay:** Fast decay suggests the system is well-represented by a few modes (low-rank).
- **Modal Alignment:** Compare learned POD modes $u_i$ with ground-truth eigenfunctions $\phi_i$ using the **Modal Assurance Criterion (MAC)** or cross-overlap matrix: $O_{ij} = \int u_i \phi_j dx$.

---

## 3. Implementation Checklist for this Project

- [x] **Quadrature Consistency:** Ensure the same integration rule (e.g., Trapezoidal) is used in the model's forward pass, the loss function, and the POD diagnostic.
- [x] **Batch-wise vs. Grid-wise:** In 1D, we usually normalize over the full spatial grid. In higher dims, ensure the "Normalization" happens over the spatial dimensions, not the batch dimension.
- [x] **Energy Ordering:** Normalization alone doesn't fix mode collapse. Pair orthonormalization with an **Energy Ordering Loss** ($E_1 < E_2 < ...$) to force the networks to pick up different eigenstates.
- [x] **Sign Ambiguity:** Wavefunctions $\psi$ and $-\psi$ are physically equivalent. POD diagnostics should align signs to a reference (e.g., forcing the maximum value to be positive) before computing errors.

## Summary Table

| Checklist Item | Description | Status | Key Source Files | Key CLI Files |
| --- | --- | --- | --- | --- |
| **Quadrature Consistency** | Uniform integration rule (Trapezoidal / Simpson) applied consistently across model forward pass, loss functions, and POD diagnostics. | Implemented | `src/utils.py`, `src/model.py`, `src/physics.py`, `src/pod.py` | `cli/cli_train.py`, `cli/extract_metrics.py` |
| **Batch-wise vs. Grid-wise** | In 1D, normalization and $L^2$ inner products integrate over the full spatial grid rather than reducing over the batch dimension. | Implemented | `src/model.py`, `src/utils.py`, `src/pod.py` | `cli/cli_train.py` |
| **Energy Ordering** | Softly penalized via `energy_ordering_loss` ($\text{ReLU}(E_i - E_{i+1})^2$) added to the training objective, preventing mode swapping while maintaining smooth gradients. | Softly Penalized | `src/physics.py`, `src/train.py` | `cli/cli_train.py` |
| **Sign Ambiguity** | Resolved via `align_modes_by_reference` in POD diagnostics, flipping modal signs based on reference overlap before metric evaluation. | Implemented | `src/pod.py`, `src/visualizations.py` | `cli/cli_train.py`, `cli/extract_metrics.py` |