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
        self._default_char_id = self._char_dict[default_char]

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
                data=rng.random(size=[1, hidden_dim]),
                dtype=torch.float32,
                requires_grad=True,
            )

        self._encoder_mlp = encoder_mlp
        self._decoder_mlp = decoder_mlp
        self._initial_hidden_state = initial_hidden_state_tensor
        self._char_vector = torch.zeros(
            size=[1, len(char_list)],
            dtype=torch.float32,
        )

    def consume(self, s):
        loss = 0
        hidden_state = self._initial_hidden_state.detach().requires_grad_()
        char_one_hot = self._get_char_one_hot(s[0])
        for c in s[1:]:
            rnn_input = torch.concatenate(
                [char_one_hot, hidden_state],
                axis=1,
            )
            hidden_state = self._encoder_mlp.forward(rnn_input)
            rnn_output = self._decoder_mlp.forward(hidden_state)
            loss += nn.loss.cross_entropy_loss(
                rnn_output,
                [self._get_char_id(c)],
            )
            char_one_hot = self._get_char_one_hot(c)

        return loss

    def _get_char_id(self, c):
        return self._char_dict.get(c, self._default_char_id)

    def _get_char_one_hot(self, c):
        char_id = self._get_char_id(c)
        self._char_vector *= 0.0
        self._char_vector[0, char_id] = 1.0
        return self._char_vector
