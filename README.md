# Design Overview
Robax is a Python package for training modern robotics models in JAX.


### Repository Structure
```
/robax
  ├── model/
  │   ├── components/
  │   │   └── attention.py, moe_transformer_block.py, etc.
  │   ├── img_model/
  │   │   └── vit.py
  │   └── policy/
  │   │   └── pi_zero.py
  │   │   └── ...
  ├── training/ (will be refactored soon...)
  │   ├── data_utils/
  │   │   └── jax_dataloader.py
  │   │   └── ...
  │   ├── objectives/
  │   │   └── flow_matching_action.py
  │   │   └── ...
  │   └── train.py, etc.
  ├── utils/
  │   └── model_visualization.py, etc.
  └── tests/
```
