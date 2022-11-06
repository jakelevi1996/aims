import torch
import numpy as np
import util
import nn

SPACE = " "
DOUBLE_SPACE = SPACE * 2

class CharRnn:
    def __init__(
        self,
        char_list,
        default_char=None,
        hidden_dim=None,
        encoder_mlp=None,
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
            hidden_dim = 5 * len(char_list)
        if encoder_mlp is None:
            encoder_mlp = nn.Mlp(
                input_dim=(len(char_list) + hidden_dim),
                output_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=2,
                output_act=nn.activation.gaussian,
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

        self._encoder_mlp = encoder_mlp
        self._decoder_mlp = decoder_mlp
        self._initial_hidden_state = torch.zeros(
            size=[1, hidden_dim],
            dtype=torch.float32,
        )
        self._char_vector = torch.zeros(
            size=[1, len(char_list)],
            dtype=torch.float32,
        )
        self._hidden_state = self._initial_hidden_state

    def consume(self, s):
        loss = 0
        self._hidden_state = self._initial_hidden_state
        char_one_hot = self._get_char_one_hot(s[0])
        for c in s[1:]:
            rnn_input = torch.concatenate(
                [char_one_hot, self._hidden_state],
                axis=1,
            )
            self._hidden_state = self._encoder_mlp.forward(rnn_input)
            rnn_output = self._decoder_mlp.forward(self._hidden_state)
            loss += nn.loss.cross_entropy_loss(
                rnn_output,
                [self._get_char_id(c)],
            )
            char_one_hot = self._get_char_one_hot(c)

        return loss / len(s)

    def cuda(self, cuda_device_id=0):
        self._cuda_device_id = cuda_device_id
        self._encoder_mlp.cuda(cuda_device_id)
        self._decoder_mlp.cuda(cuda_device_id)
        self._char_vector = self._char_vector.cuda(cuda_device_id)
        self._initial_hidden_state = (
            self._initial_hidden_state.cuda(cuda_device_id)
        )

    def get_params(self):
        return self._encoder_mlp.get_params() + self._decoder_mlp.get_params()

    def zero_grad(self):
        self._encoder_mlp.zero_grad()
        self._decoder_mlp.zero_grad()

    def train(
        self,
        data_str,
        optimiser,
        batch_size=64,
        max_num_batches=int(1e5),
        max_num_seconds=(5 * 60),
        print_every=10,
        predict_every=100,
        predict_args=None,
    ):
        loss_list = []
        time_list = []
        timer = util.Timer()
        s_ptr = 0
        batch_ind = 0
        for batch_ind in range(max_num_batches):

            s_batch = data_str[s_ptr:(s_ptr + batch_size)]
            while DOUBLE_SPACE in s_batch:
                s_batch = s_batch.replace(DOUBLE_SPACE, SPACE)

            loss_tensor = self.consume(s_batch)
            loss_tensor.backward()
            optimiser.step()
            self.zero_grad()

            loss = loss_tensor.item()
            loss_list.append(loss)
            time_list.append(timer.time_taken())
            s_ptr += batch_size
            if s_ptr >= len(data_str):
                s_ptr = 0
            if (batch_ind % print_every) == 0:
                print(
                    "Batch %4i | Loss = %.3f | " % (batch_ind, loss),
                    end="",
                )
                timer.print_time_taken()
            if predict_args is not None:
                if (batch_ind % predict_every) == 0:
                    self.predict(*predict_args)
            if timer.time_taken() >= max_num_seconds:
                break

            batch_ind += 1

        return time_list, loss_list

    def predict(self, prompt, num_chars=500, print_each_char=True):
        if print_each_char:
            print(prompt, end="", flush=True)

        self.consume(prompt)
        char_pred_list = []
        for _ in range(num_chars):
            rnn_output = self._decoder_mlp.forward(self._hidden_state)
            char_pred = self._char_list[torch.argmax(rnn_output).item()]
            char_one_hot = self._get_char_one_hot(char_pred)
            rnn_input = torch.concatenate(
                [char_one_hot, self._hidden_state],
                axis=1,
            )
            self._hidden_state = self._encoder_mlp.forward(rnn_input)

            char_pred_list.append(char_pred)
            if print_each_char:
                print(char_pred, end="", flush=True)

        if print_each_char:
            print()

        return prompt + "".join(char_pred_list)

    def _get_char_id(self, c):
        return self._char_dict.get(c, self._default_char_id)

    def _get_char_one_hot(self, c):
        char_id = self._get_char_id(c)
        self._char_vector *= 0.0
        self._char_vector[0, char_id] = 1.0
        return self._char_vector
