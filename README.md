# Design Overview
### Repository Structure
```
/pi_zero_project
  ├── model/
  │   ├── components/
  │   │   └── attention.py, moe_transformer_block.py, etc.
  │   ├── vlm/
  │   │   └── img_model/
  │   │   └── llm_model/ (can be removed)
  │   └── policy/
  │   │   └── ...
  ├── training/
  │   ├── data_utils/
  │   │   └── jax_dataloader.py
  │   │   └── ...
  │   ├── objectives/
  │   │   └── ...
  │   └── train.py, etc.
  ├── utils/
  │   └── model_visualization.py, etc.
  └── tests/
```

I started off by porting over the ViT, Gemma, and PaliGemma code from Google's [Big Vision repository](https://github.com/google-research/big_vision). While I refactored much of it, there still exist remnants of the original experimentation that I did from the Big Vision codebase.

### Overview of the Pi_Zero Implementation
The model is defined in `model/policy/pi_zero.py`. 

The `__call__` function of the `PiZero` module has the following arguments:
- `images`: (batch_size, num_images, height, width, 3)
- `text`: (batch_size, num_text_tokens)
- `proprio`: (batch_size, num_proprio_states, num_proprio_features)
- `action`: (batch_size, num_actions, num_action_features)
- `timesteps`: (batch_size,)

And it returns a (batch_size, num_actions, num_action_features) prediction and a dictionary of intermediate values. During execution, it begins by embedding all input modalities before performing a joint attention operation. Following the paper, different tokens are handled by different experts, and a custom mask is used to ensure blockwise causal attention accross the image + text, proprioceptive, and action modalities. 

![architecture](assets/architecture.png)

Additionally, flow matching loss is implemented in `training/objectives/flow_matching_action.py`. Most of the code is well documented and should hopefully be easy to parse.



### Mixture of Experts Implementation
Most of the mixture of experts logic occurs in the `MoEAttention` class, which is defined in `model/components/attention.py`. Essentially, the model performs attention with the following mask:

```
--gemma-- --action expert--
img + txt    prop   act   ]
[i, i, t, t, p, p, a, a, a]

[1, 1, 1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 0, 0, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 0, 0, 0]
[1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1]
[1, 1, 1, 1, 1, 1, 1, 1, 1]
```

When designing the model, I was primarily deciding between two implementation options:
1. Run PaliGemma first while caching the KV values before running the attention expert using the cached values.
2. Calculate QKV for each expert at each layer before concatenating and running a single attention operation.

While the first option allows us to easily reuse PaliGemma, it loses the ability to run the attention operation in parallel, the cache eats memory during training, and it's harder to adapt to new action generation formulations.

The second option (which is implemented in the codebase) is more flexible and allows us to run the attention operation in parallel. If designed well, it also allows us to mix and match experts accross many attention mask formulations.

### Training on Push-T
In order to train on the `Push-T` dataset, I wrote a naive wrapper around the LeRobot 2.0 dataset class. I chose `Push-T` becuase I thought it'd provide a simple validation of the model, and I used LeRobot because it provides a pretty seemless way to load small robotics datasets. This is defined in `training/data_utils/jax_lerobot_dataset.py`. I also implemented a simple `train.py` script that can be used to train the model.

The model is currently being trained on the `Push-T` dataset. The loss curve so far is shown below:
![Loss Curve](assets/loss_curve.png)

The initial drop seems dubiously steep, and I'd have to continue to tune / debug the model as I validate performance in simulation.
