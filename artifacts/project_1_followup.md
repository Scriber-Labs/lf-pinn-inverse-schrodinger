# Symplectic Structures and the Inverse Schrödinger Problem
## Setting the Geometric Stage with Classical Mechanics
> 💡 The inverse Schrödinger problem can be cleanly mapped to learning a scalar Hamiltonian whose associated bivector flow preserves the canonical symplectic form $dq \wedge dp$. 

Let phase space be defined mathematically as $(q,p)\in T^*Q$. 
- ❓❓❓what are matrices T and Q supposed to be?

This allows us to further define $\omega= dq \wedge dp$ as the canonical symplectic form. Importantly, this is a 2-form that:
- defines what dynamics are allowed
- is *prior* to the Hamiltonian
- is the structure quantization preserves

## Quantization
1. $q \rightarrow \hat{x}$
2. $p \rightarrow -i\hbar \partial_x$

## Schrödinger Operator as a Symplectic Structure
Recall: $\hat{H}=-\frac{\hbar^2}{2m}\partial_x^2+V(x)$ is the Hamiltonian operator used in the TISE.

Symplectic structure mapping: 

$[q,p]=i\hbar \quad \leftrightarrow \quad \omega=dq\wedge dp$

> 🏡 Enforcing the Schrödinger equation as a hard constraint by default enforces a quantized symplectic geometry.

## Loss function Residual and The canonical symplectic form
Schrodinger loss residual used in the repo:
$$L_\text{SE} = \bigg|\bigg| \bigg(  -\frac{\hbar^2}{2m}\psi'' + V_\theta\psi - E\psi  \bigg) \bigg|\bigg|^2$$

> 🗝️ Since $\psi'' \approx p^2\psi$, the residual enforces the correct quadratic form induced by the symplectic metric. (❓)
>  Specifically,
>  - $dq$ acts on functions (❓)
>  - $dp$ acts asa differentiation (❓)
>  - the wedge structure allows second-order operators (❓)

### Justification for extra structure
Project 1: symplectic structure and Hamiltonian gave a well-posed problem.

Project 2: the inverse Schrodinger problem is ill-posed due to the fact that many possible potential functions can give rise to the same eigenstates. (❓I NEED THIS STATEMENT TO BE FIXED AND EXPLAINED)
- Regularization introduces geometric bias so that the hypothesis space is constrained.
  - Introducing $$\int(V''(x))^2dx$$ penalizes curvature in the configuration-space projection of phase space (i.e., wild folding of Lagrangian submanifolds). (❓❓❓❓WHAT DOES THIS MEAN? IT SOUNDS IMPORTANT!)

> 🏡 The smoothness term regularizes the inverse problem by discouraging rapid curvature of the effective Lagrangian submanifold induced by $V(x)$.

## Clifford algebra detour
Importantly, Clifford algebra does not introduce additional parameters or structure. It is the linearlization of symplectic geometry (❓i would really like a source for this that i can look at for myself).

For the sake of keeping it simple, we start with establishing 1D phase space:
- basis: $\{ \mathbb{e}_q, \mathbb{e}_p \}$d
- geometric product: $$\mathbb{e}_q \mathbb{e}_p = \mathbb{e}_q\cdot\mathbb{e}_p+\mathbb{e}_q\wedge\mathbb{e}_p$$
  - This is a good place to distinguish $dq$ and $dp$ from $\mathbb{e}_q$ and $\mathbb{e}_p$. If there is no real distinction, I need to decide how i want to define my basis for this sort of context from here on out. or at least have a game plan since this will come up a lot. 

## Symplectic Consistency Loss
Let $$p_\psi(x) := -i\hbar \frac{\psi'(x)}{\psi(x)}$$ be the **quantum momentum field**. Then define $$\mathcal{L}_\text{symp}=\bigg|\bigg| \frac{dp_\psi}{dx} + V'(x) \bigg|\bigg|^2$$

---
Project 1 questions that said i'd address in this project(at least conceptually review for myself).
 - [ ] Vectors vs. covectors (and why momenta live naturally as covectors)
 - [ ] Covectors in geometric/ Clifford algebra
 - [ ] Hamiltonian mechanics from geometry (kinetic + potential 
→
 flow)
 - [x] Laplacian from Hamiltonian structure
 - [ ] Duals of covectors (contrast with cross vs. wedge products)
 - [ ] Convince myself that J = [0 1; -1 0] is the matrix representation of the canonical symplectic form.
 - [ ] Bivector contractions and Poisson brackets.
-- 
