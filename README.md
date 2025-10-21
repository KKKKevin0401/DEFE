# Dynamic Nonlinear Networks for Adaptive Low-Light Image Enhancement

> **Abstract:** *While diffusion-based methods have advanced low-light image enhancement, they face two critical limitations: the tendency to suppress complex high-frequency details by misinterpreting them as noise, and persistent artifacts such as insufficient denoising and color shifts. To address these challenges, we propose a novel framework, Dynamic Nonlinear Networks, inspired by the Kolmogorov-Arnold representation theorem. Our framework introduces a Dynamic Edge Feature Extractor (DEFE), which leverages learnable nonlinear convolutions and adaptive residual connections to reconstruct high-frequency features prior to the diffusion process, effectively preventing the diffusion process from erroneously suppressing fine textures. To further enforce structural fidelity, we introduce a dedicated edge-preserving loss. Furthermore, to tackle residual artifacts, we leverage a Residual Refinement Module  that is conditioned on the original low-light input. This mechanism explicitly couples noise estimation with illumination degradation, enabling precise correction of noise and color deviations without sacrificing detail through uniform smoothing. Extensive experiments on benchmark datasets, including LOL and SICE, demonstrate that our method achieves state-of-the-art performance, particularly in preserving intricate details and ensuring color fidelity.* 
>

<p align="center">
  <img width="800" src="figs/pipeline.jpg">
</p>

---
