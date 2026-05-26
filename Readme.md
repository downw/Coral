# CORAL: Uncertainty-Aware Regulation of Exposure Concentration in Recommender Systems

## Abstract
Recommender systems (RS) may suffer from feedback-driven exposure concentration, where repeated engagement optimization collapses exposure onto a narrow subset of categories, shrinking catalog coverage and undermining long-horizon learning. While prior methods reduce concentration empirically, they are often post hoc and typically lack principled uncertainty-aware risk estimates for regulating exposure under endogenous feedback. To address this, we formulate exposure regulation as a constrained sequential decision problem, i.e., maximize recommendation utility while limiting saturation violations. Specifically, we propose CORAL, a model-agnostic, uncertainty-aware framework for recommendation that regulates exposure dynamics under endogenous feedback. It first models self-reinforcing interactions with a Hawkes-inspired intensity model to construct an exposure saturation state. It then derives an upper confidence bound on the category-conditioned risk of violating the saturation threshold from the observed violation history, and integrates this bound into the decision rule using a state-dependent penalty for adaptive intervention near saturation. We provide theoretical results establishing risk bounds, finite-time recovery, and efficient long-term performance. Empirical results validate CORAL achieves competitive utility with reduced exposure saturation and improved long-run stability on real-world datasets and controlled simulations.

## Table of Contents

- [Abstract](#abstract)
- [Key Features](#key-features)
- [Datasets Preprocessing](#datasets-preprocessing)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)


## Key Features
✅ Model-Agnostic: Works with any sequential recommendation backbone (SASRec, BERT4Rec, etc.)

✅ Uncertainty-Aware: Derives confidence bounds on saturation violation risk

✅ Adaptive Intervention: Applies minimal regulation when safe, increases only near saturation

✅ Theoretical Guarantees: Proven finite-time recovery and sublinear regret

✅ Production-Ready: Efficient online inference with low computational overhead

```
┌─────────────────────────────────────────────────────────────┐
│                      CORAL Framework                         │
├─────────────────────────────────────────────────────────────┤
│  1. Hawkes-Inspired Intensity Model                          │
│     └─ Captures self-reinforcing exposure dynamics           │
│                                                               │
│  2. Exposure Saturation State                                │
│     └─ Dt = Σ wc · max(0, λu,c(t) - ρ·μu,c)                 │
│                                                               │
│  3. Risk Estimation with Confidence Bounds                   │
│     └─ R̂⁺t(c) = empirical risk + uncertainty term           │
│                                                               │
│  4. Adaptive State-Dependent Penalty                         │
│     └─ Λ(Dt) = min{Λmax, 1/max(ε, λmax - Dt)}              │
│                                                               │
│  5. Risk-Aware Category Selection                            │
│     └─ ct = argmax[r̂t(c) + κ√(ln t/Nt(c)) - Λ(Dt)R̂⁺t(c)]  │
└─────────────────────────────────────────────────────────────┘
```
## <a id="datasets-preprocessing"></a>📁 datasets preprocessing

Original dataset download links are available in the paper.

First, run <dataset_name>/Core_filter.ipynb.
Then, run the processing script for the specific dataset:
```bash
python <dataset_name>/processing_Amazon.py
```
The processed data will be saved in a processed subfolder.

## <a id="quick-start"></a>🚀 Quick Start

prepare environment
```bash
pip install -r requirements.txt
```

static offline evaluation

```bash
python static_offline/main.py --dataset Amazon --lambda_max 0.7 --kappa 2 --delta_conf 0.1 
python static_offline/main.py --dataset ML1m --lambda_max 0.7 --kappa 2 --delta_conf 0.1 
python static_offline/main.py --dataset Steam --lambda_max 0.9 --kappa 2 --delta_conf 0.05 
```
online closed-loop evaluation
To simulate realistic user behavior, we employ the Gemma-3-12B-IT large language model for online closed-loop evaluation
```bash
chmod +x run_parallel_simulation.sh
./run_parallel_simulation.sh
```

To use a different backbone (e.g., BERT4Rec), replace model.py with model_BERT4Rec.py.

## <a id="project-structure"></a>📁 Project Structure

├── datasets/                # Data processing and storage directory
│   ├── Amazon/              
│   ├── MovieLen/            
│   └── Steam/               
├── figs/                    # Directory for generated figures
└── src/                     # Main source code
    ├── baselines/           # Baseline algorithms and comparison models
    │   ├── Allostatic/     
    │   ├── HardFilter/      
    │   ├── Nuge/           
    │   ├── ReinforceBreak/  
    │   └── TD-VAE-CF/       
    ├── model4Sim/           # online closed-loop evaluation
    └── static_offline/      # offline evaluation