This document provides high level overviews of the model implementations in the `robax` package.

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
