import torch
import numpy as np
import nn

class CharRnn:
    def __init__(
        self,
        char_list,
        default_char=None,
        hidden_dim=None,
        encoder_mlp=None,
        decoder_mlp=None,
        initial_hidden_state_tensor=None,
        rng=None,
    ):
        self._char_list = char_list
        self._char_dict = {c: i for i, c in enumerate(char_list)}
        if default_char is None:
            default_char = char_list[0]
        self._default_char_ind = self._char_dict[default_char]

        if rng is None:
            rng = np.random.default_rng(0)

        if hidden_dim is None:
            if initial_hidden_state_tensor is not None:
                hidden_dim = initial_hidden_state_tensor.numel()
            else:
                hidden_dim = 5 * len(char_list)
        if encoder_mlp is None:
            encoder_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=2,
                output_act=nn.activation.linear,
                hidden_act=nn.activation.relu,
            )
        if decoder_mlp is None:
            decoder_mlp = nn.Mlp(
                input_dim=hidden_dim,
                output_dim=len(char_list),
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=nn.activation.linear,
                hidden_act=nn.activation.relu,
            )
        if initial_hidden_state_tensor is None:
            initial_hidden_state_tensor = torch.tensor(
                rng.random(size=hidden_dim),
                dtype=torch.float32,
                requires_grad=True,
            )
