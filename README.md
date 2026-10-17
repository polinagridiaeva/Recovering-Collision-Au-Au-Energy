# Machine Learning for Energy Reconstruction in Heavy-Ion Collisions

Bachelor's thesis project, Saint Petersburg State University (Department of Computational Physics, 2025).
Supervisor: Prof. V. A. Rudnev.

This repository contains the code for estimating, event by event, the energy released in Au+Au collisions at $\sqrt{s_{NN}} = 11$ GeV from the signal of two microchannel-plate (MCP) detectors, using neural networks. The work was motivated by the possible installation of such detectors at the MPD experiment at NICA (JINR, Dubna).

## Overview

Not all collision parameters can be measured directly. The energy released in a collision has to be inferred from what the detectors record: which detector cell was hit and when. In this project:

1. Au+Au collisions were simulated with the QGSM event generator (200,000 events).
2. Energy and momentum conservation were checked in the generated data: energy is conserved within 1.5%, momentum within 14%.
3. Spectator nucleons, which do not take part in the interaction, were removed using energy and momentum filters. The released energy was then studied as a function of the impact parameter $b$, and its dependence $\sqrt{s_{\text{eff}}}(b)$ agrees with the expected geometric behaviour.
4. A neural network was trained to map the detector signal directly to the released energy.

## Detector model and input features

- Two ring-shaped MCP detectors placed at $z = \pm 400$ cm from the collision point (outer diameter 50 cm, inner 5 cm).
- Each detector is segmented into 6 rings × 10 sectors = 60 pads; each pad registers only the first particle that reaches it.
- Particle trajectories are approximated as straight lines from the collision point to the detector plane.

For each registered particle the network receives three features:

| Feature | Description |
|---|---|
| $T = z_0 / (\mathrm{tof} - t_0)$ | Transformed time of flight ($t_0$ = time for light to reach the detector) |
| Pad number | Ring and sector of the hit pad |
| Detector ID | Which of the two detectors was hit |

With at most 120 hits (2 detectors × 60 pads), every event is encoded as a 3 × 120 feature matrix (360 values).

## Models

Both networks are implemented in PyTorch and trained with MSE loss and the Adam optimizer (learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 3000, up to 1500 epochs with early stopping, 80/20 train/test split).

- Fully connected NN: 360 → 128 → 64 → 32 → 16 → 1, SiLU activations.
- Hybrid NN: a Conv2D layer (8 filters, 6 × 3 kernel, ReLU) applied to the 120 × 3 input, followed by the same fully connected head.

Each model was trained in two settings: using all particles, and using only charged particles (what an MCP detector can actually register).

## Results

RMSE of the reconstructed energy on the test set, and relative error $\varepsilon = \mathrm{RMSE} / \sqrt{s_{\text{total}}}$, with $\sqrt{s_{\text{total}}} = 2168$ GeV:

| | Conv2D + FC | Fully connected |
|---|---|---|
| All particles | 150.5 GeV (6.95%) | 154.0 GeV (7.10%) |
| Charged particles | 203.3 GeV (9.38%) | 211.6 GeV (9.77%) |

- The relative error stays below 10% in all configurations, even though the detectors cover only a small solid angle.
- Adding a convolutional layer consistently lowers the error.
- In the mid-energy range the prediction errors are close to Gaussian. At low and high energies the error distributions are skewed, because the model does not know that the released energy must lie between 0 and $\sqrt{s_{\text{total}}}$.

## Repository structure

```
├── released_energy_qgsm_NNs.ipynb   # Full pipeline: feature extraction, models, training, evaluation
├── vkr.pdf                          # Bachelor's thesis (original, in Russian)
├── thesis_en.pdf                    # Bachelor's thesis (English translation)
├── english_version.pdf              # Presentation
├── abstracts.pdf                    # Conference abstracts
└── README.md
```

The notebook covers:
- geometry of the detector and projection of particle trajectories onto the detector plane;
- assignment of hits to pads and construction of the feature vector;
- spectator filtering and computation of the released energy;
- the fully connected and hybrid (Conv2D) regressors and the training loop;
- evaluation: RMSE, predicted-vs-true plots, and error histograms with Gaussian fits for low, mid and high energies.

## Data

The QGSM-simulated events (`data_*.dat`, 100 files × 2000 events) are not included in this repository because of their size. To run the notebook, update the data path in the loading cell.

## Requirements

Python 3 with `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `torch`.

## Possible improvements

- Normalising inputs and targets to $[0, 1]$.
- Predicting energy per nucleon–nucleon pair instead of the total.
- Deeper convolutional architectures.
- A loss function that accounts for the physical bounds on the released energy and the resulting asymmetry of the errors.

## Copyright Notice

© 2026 [Polina Gridiaeva]. All rights reserved.

This repository contains research code associated with an academic thesis and related future publications.

The source code, documentation, and any associated materials in this repository may not be copied, modified, distributed, published or used in whole or in part without explicit prior written permission from the author.





