import torch
import numpy as np
import util
import nn

SPACE = " "
DOUBLE_SPACE = SPACE * 2

class CharLstm(nn.CharRnn):
    def __init__(
        self,
        char_list,
        default_char=None,
        hidden_dim=None,
        cell_dim=None,
        forget_mlp=None,
        input_mlp=None,
        output_mlp=None,
        input_select_mlp=None,
        output_select_mlp=None,
        decoder_mlp=None,
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
            hidden_dim = 3 * len(char_list)
        if cell_dim is None:
            cell_dim = len(char_list)
        if forget_mlp is None:
            forget_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=cell_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=nn.activation.sigmoid,
                hidden_act=nn.activation.relu,
                rng=rng,
            )
        if input_mlp is None:
            input_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=cell_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=torch.tanh,
                hidden_act=nn.activation.relu,
                rng=rng,
            )
        if output_mlp is None:
            output_mlp = nn.Mlp(
                input_dim=cell_dim,
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=torch.tanh,
                hidden_act=nn.activation.relu,
                rng=rng,
            )
        if input_select_mlp is None:
            input_select_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=cell_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=nn.activation.sigmoid,
                hidden_act=nn.activation.relu,
                rng=rng,
            )
        if output_select_mlp is None:
            output_select_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=nn.activation.sigmoid,
                hidden_act=nn.activation.relu,
                rng=rng,
            )
        if decoder_mlp is None:
            decoder_mlp = nn.Mlp(
                input_dim=hidden_dim,
                output_dim=len(char_list),
                hidden_dim=hidden_dim,
                num_hidden_layers=1,
                output_act=nn.activation.linear,
                hidden_act=nn.activation.relu,
                rng=rng,
            )

        self._forget_mlp        = forget_mlp
        self._input_mlp         = input_mlp
        self._output_mlp        = output_mlp
        self._input_select_mlp  = input_select_mlp
        self._output_select_mlp = output_select_mlp
        self._decoder_mlp       = decoder_mlp
        self._char_vector = torch.zeros(
            size=[1, len(char_list)],
            dtype=torch.float32,
        )
        self._initial_hidden_state = torch.zeros(
            size=[1, hidden_dim],
            dtype=torch.float32,
        )
        self._initial_cell_state = torch.zeros(
            size=[1, cell_dim],
            dtype=torch.float32,
        )
        self._hidden_state  = self._initial_hidden_state
        self._cell_state    = self._initial_cell_state

    def _initialise_hidden_state(self):
        self._hidden_state  = self._initial_hidden_state
        self._cell_state    = self._initial_cell_state

    def _update_hidden_state(self, rnn_input):
        self._cell_state = (
            (self._cell_state * self._forget_mlp.forward(rnn_input))
            + (
                self._input_mlp.forward(rnn_input)
                * self._input_select_mlp.forward(rnn_input)
            )
        )
        self._hidden_state = (
            self._output_mlp.forward(self._cell_state)
            * self._output_select_mlp.forward(rnn_input)
        )

    def cuda(self, cuda_device_id=0):
        self._cuda_device_id = cuda_device_id
        self._forget_mlp.cuda(cuda_device_id)
        self._input_mlp.cuda(cuda_device_id)
        self._output_mlp.cuda(cuda_device_id)
        self._input_select_mlp.cuda(cuda_device_id)
        self._output_select_mlp.cuda(cuda_device_id)
        self._decoder_mlp.cuda(cuda_device_id)
        self._char_vector   = self._char_vector.cuda(cuda_device_id)
        self._hidden_state  = self._hidden_state.cuda(cuda_device_id)
        self._cell_state    = self._cell_state.cuda(cuda_device_id)
        self._initial_hidden_state = (
            self._initial_hidden_state.cuda(cuda_device_id)
        )
        self._initial_cell_state = (
            self._initial_hidden_state.cuda(cuda_device_id)
        )

    def get_params(self):
        params = (
            self._forget_mlp.get_params()
            + self._input_mlp.get_params()
            + self._output_mlp.get_params()
            + self._input_select_mlp.get_params()
            + self._output_select_mlp.get_params()
            + self._decoder_mlp.get_params()
        )
        return params

    def zero_grad(self):
        self._forget_mlp.zero_grad()
        self._input_mlp.zero_grad()
        self._output_mlp.zero_grad()
        self._input_select_mlp.zero_grad()
        self._output_select_mlp.zero_grad()
        self._decoder_mlp.zero_grad()
