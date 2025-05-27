# Layer-wise Relevance Propagation for Layer Normalized LSTMs

## 📖 Overview

Welcome to the codebase for my Master’s thesis and internship project at the IBM Watson Center in Munich. This was the first “big” code project I carried out, and lead to:

- A live demonstrator showcased in the IBM showroom  
- My first peer-reviewed publication (IEA/AIE 2022):
  
  > Wehner, C., Powlesland, F., Altakrouri, B., & Schmid, U. (2022, July). _Explainable online lane change predictions on a digital twin with a layer normalized LSTM and layer-wise relevance propagation_. In _International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems_ (pp. 621–632). Cham: Springer International Publishing.
   
  [View on Google Scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=eU7sl_kAAAAJ&citation_for_view=eU7sl_kAAAAJ:u-x6o8ySG0sC)  
 

Here I explore how to **predict** and **explain** lane-change maneuvers on a digital twin of a German autobahn using:

- A **layer‐normalized LSTM** for time-series lane-change prediction  
- **Layer-wise Relevance Propagation (LRP)** (including a novel Ω-rule) for post-hoc explainability  
- A Dockerized Flask API that performs live inference and serves human-readable explanations  

I learned a ton of foundational lessons in architecture, clean code, and tooling—many of which I’ve since improved upon in later projects. Nevertheless, I’m proud of this work and happy to share the core pieces that are safe for public release.

---

## 🗂 Repository Structure

The code is organized into three main parts:

### 1. `layer_normalized_lstm/`

- **Data preprocessing**  
  - Scripts to ingest and convert the Providentia++ autobahn protobuf stream into TensorFlow-ready datasets.  
- **Model training & evaluation**  
  - A TensorFlow 2 implementation of a layer-normalized LSTM, training and evaluation.  

### 2. `xai-prediction-engine/`

This is the heart of the demonstrator:

- **Dockerized Flask API**  
  - Exposes a `/snaps` endpoint that accepts a JSON “snapshot” of the autobahn state.  
  - Returns both a lane-change predication and its LRP-based explanation.  
- **LSTM + full LRP backward pass**  
  - `lstm_layer_norm_network.py` fully re-implements the layer-normalized LSTM in base-level TensorFlow.  
  - Contains a the implementation of an LRP backward-pass, including our custom Ω-rule for more faithful relevance scores.   

### 3. `attribution_methods/`

Here, I explore and benchmark alternative explainability approaches:

- **Baseline attribution methods**  
  - Implementations of Integrated Gradients, Input×Gradient, and other XAI techniques.  
- **Comparative evaluation**  
  - `perturbation_test.py` runs perturbation (faithfulness) tests, reproducing the figures and metrics from our paper.  
---


## 📄 Bibliography

```bibtex
@inproceedings{wehner2022explainable,
  title={Explainable online lane change predictions on a digital twin with a layer normalized lstm and layer-wise relevance propagation},
  author={Wehner, Christoph and Powlesland, Francis and Altakrouri, Bashar and Schmid, Ute},
  booktitle={International Conference on Industrial, Engineering and Other Applications of Applied Intelligent Systems},
  pages={621--632},
  year={2022},
  organization={Springer}
}
```