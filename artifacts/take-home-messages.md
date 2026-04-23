# Project 2 Notes

## 🌊 Five Brunton Steps

| **Step**               | **Description**                                                                                                                                                                        | **Completed?**                                                                                                                                                      |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Problem Formulation | Can a physics-informed neural network recover the unknown potential $V(x)$ along with the corresponding eigenfunctions from only noisy spectral data and probability-density snapshots? | ✔️                                                                                                                                                                  |
| 2. Data Collection & Curation | - Uniform collocation grid of spatial points $x\in[-5,5]$ <br/> - Noisy observations: approximate energy levels and probability densities with Gaussian noise                          | ✔️                                                                                                                                                                  |
| 3. Neural Architecture | Two lightwieght neural networks with tanh activations (see architecture diagram)                                                                                                       | ⚠️ Note that both NNS are scarlar-in, scalar-out, fully differentiable, and deliberately kept shallow to preserve interpretibility️                                 |
| 4. Loss Function | All terms are soft penalties with static $\lambda$ weights                                                                                                                             | ✔️                                                                                                                                                                  |
| 5. Optimization Strategy | - Optimizer: Adam with a fixed learning rate <br/> - Training loop: forward pass; compute loss terms; back propagate; update $V_\theta$ and $\psi_n^\theta$                            | ❌⚠️ As in project 1, the optimizer is intentionally vanilla; the aim is to expose how the physics prior interacts with noisy data, not to chace maximal performance |                                                                                                                       |
- 

## 🏡 Take-Home Messages

1. Don't carelessly introduce normalization steps without keeping track. And if you double normalize, make note of it and justify it.
    - Q: Is double-normalization  bad?
    - A: Usually,
      - numerically harmless
      - but conceptually messy
      - ⚠️ Can mask whether the model itself is enforcing normalization or whether the training loop is doing it.
2. 
