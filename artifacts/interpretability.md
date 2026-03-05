# 🧠 Interpretability Axis

## 💡Ideas:

Three interpretability layers (Figure 6 analysis so far):

| **Layer** | **Diagnostic** |
|-----------|----------------|
| Observable level | density matching |
| Geometric level | node/ parity structure |
| Operator level | orthogonality & spectrum |

---
For the nondimensionalized quantum harmonic oscillator (QHO), eigenstates are eigenfunctions of:

$$ H = -\frac{1}{2}\frac{\partial^2}{
\partial x^2}+ \frac{1}{2}x^2$$

- ⚠️CHECK THIS FOR YOURSELF!⚠️
  - According to chat-gpt, the curvature of the QHO eigenfunctions is related to energy level where 

    $$ \psi_n \sim (x^2 - 2n -1)\psi_n $$
  $\Rightarrow$ Curvature increases with energy level!
    - Provides a natural interpretability axis that relates...
      - Laplacian energy
      - POD singular values
      - Mode curvature
      - Spectral index