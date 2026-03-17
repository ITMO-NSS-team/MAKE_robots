# MAKE_robots

---

![License](https://img.shields.io/github/license/ITMO-NSS-team/MAKE_robots?style=flat&logo=opensourceinitiative&logoColor=white&color=blue)
[![OSA-improved](https://img.shields.io/badge/improved%20by-OSA-yellow)](https://github.com/aimclub/OSA)

---

## Overview

The MAKE_robots project advances the modeling of active matter by employing a data-driven approach to uncover non-linear governing equations. It introduces the Evolutionary Partial Differential Equation (EPDE) framework, integrated with the TEDEouS solver, to derive stochastic differential equations that capture the complex dynamics of Swarmodroid robots. By transitioning from individual agent dynamics to continuum field theories, the project provides insights into both microscopic and macroscopic behaviors. This methodology, validated through the Lotka-Volterra model, offers a robust tool for understanding active matter systems, highlighting the role of velocity-dependent and Rayleigh friction terms in swarm dynamics.

This work implements the EPDE+SAGE methodology for discovering interpretable stochastic differential equations from experimental or simulated active matter data. The approach bridges discrete agent-based descriptions to continuum field theories through three hierarchical modeling levels:
- **Microscopic** — Individual agent dynamics via ODEs with interaction forces  
  `d²rₖ/dt² = Fᵢₙₜ + Σ Fₑₓₜ`
- **Mesoscopic** — Collective order parameters via coupled SDEs  
  `dΦ = A(Φ)dt + B(Φ)∘dWₜ`
- **Macroscopic** — Continuum field evolution via SPDEs  
  `∂u/∂t = ℒ_μ[u] + ℒ_σ[u]`

---

## Table of Contents

- [Overview](#overview)
- [Content](#content)
- [Algorithms](#algorithms)
- [Installation](#installation)
- [License](#license)
- [Citation](#citation)

---

## Content

The MAKE_robots project employs the EPDE+SAGE methodology to model active matter using robotic swarm data. It operates across three hierarchical levels: microscopic, mesoscopic, and macroscopic. At the microscopic level, individual agent dynamics are captured using ordinary differential equations, focusing on interaction forces. The mesoscopic level models collective behaviors through coupled stochastic differential equations, while the macroscopic level describes continuum field evolution using stochastic partial differential equations. This structured approach transitions from agent-based models to field theories, enabling the discovery of differential equations that elucidate the dynamics of active matter systems. The project integrates data processing, equation discovery, and analysis to provide a comprehensive framework for understanding complex systems.

---

## Algorithms

The project employs the Evolutionary Partial Differential Equation (EPDE) framework integrated with the TEDEouS solver to discover non-linear governing equations for active matter systems. This approach is pivotal in modeling the dynamics of Swarmodroid robots by deriving partial and stochastic differential equations (SDEs) from data. The SAGE framework reframes equation discovery as a probabilistic task, capturing uncertainties in system dynamics through symbolic SDEs. These methodologies transition from individual agent dynamics to continuum field theories, using kriging methods to model displacement fields. The framework effectively uncovers the underlying physical laws governing active matter, offering insights into microscopic and macroscopic dynamics, including velocity-dependent and Rayleigh friction terms.

---

## Installation

Install MAKE_robots using one of the following methods:

**Build from source:**

1. Clone the MAKE_robots repository:
```sh
git clone https://github.com/ITMO-NSS-team/MAKE_robots
```

2. Navigate to the project directory:
```sh
cd MAKE_robots
```

---

## License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://github.com/ITMO-NSS-team/MAKE_robots/tree/main/LICENSE) file.

---

## Citation

If you use this software, please cite it as below.

### APA format:

    ITMO-NSS-team (2026). MAKE_robots repository [Computer software]. https://github.com/ITMO-NSS-team/MAKE_robots

### BibTeX format:

    @misc{MAKE_robots,

        author = {ITMO-NSS-team},

        title = {MAKE_robots repository},

        year = {2026},

        publisher = {github.com},

        journal = {github.com repository},

        howpublished = {\url{https://github.com/ITMO-NSS-team/MAKE_robots.git}},

        url = {https://github.com/ITMO-NSS-team/MAKE_robots.git}

    }

---
