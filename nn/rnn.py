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
        self._rng = rng

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
        self._one_hot_char_tensor = torch.zeros(
            size=[1, len(char_list)],
            dtype=torch.float32,
        )
        self._hidden_state = self._initial_hidden_state
        self._cuda_device_id = None

    def _initialise_hidden_state(self, batch_size):
        prev_batch_size, hidden_dim = self._initial_hidden_state.shape
        if prev_batch_size != batch_size:
            self._initial_hidden_state = torch.zeros(
                size=[batch_size, hidden_dim],
                dtype=torch.float32,
                device=self._initial_hidden_state.device,
            )

        self._hidden_state = self._initial_hidden_state

    def _update_hidden_state(self, rnn_input):
        self._hidden_state = self._encoder_mlp.forward(rnn_input)

    def consume(self, *batch_strings):
        loss = 0
        initial_chars = [s[0] for s in batch_strings]
        one_hot_tensor = self._get_one_hot_char_tensor(*initial_chars)
        batch_size = len(batch_strings)
        self._initialise_hidden_state(batch_size)
        batch_str_len = max(len(s) for s in batch_strings)
        for char_ind in range(1, batch_str_len):
            rnn_input = torch.cat(
                [one_hot_tensor, self._hidden_state],
                axis=1,
            )
            self._update_hidden_state(rnn_input)
            rnn_output = self._decoder_mlp.forward(self._hidden_state)
            char_list = [s[char_ind] for s in batch_strings]
            loss += nn.loss.cross_entropy_loss(
                rnn_output,
                [self._get_char_id(c) for c in char_list],
            )
            one_hot_tensor = self._get_one_hot_char_tensor(*char_list)

        return loss / batch_str_len

    def cuda(self, cuda_device_id=0):
        self._cuda_device_id = cuda_device_id
        self._encoder_mlp.cuda(cuda_device_id)
        self._decoder_mlp.cuda(cuda_device_id)
        self._hidden_state = self._hidden_state.cuda(cuda_device_id)
        self._one_hot_char_tensor = (
            self._one_hot_char_tensor.cuda(cuda_device_id)
        )
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
        batch_size=100,
        batch_str_len=64,
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
        for batch_ind in range(max_num_batches):

            batch_str_list = []
            for _ in range(batch_size):
                s_ptr_batch_end = s_ptr + batch_str_len
                while True:
                    s_batch = data_str[s_ptr:s_ptr_batch_end]
                    while DOUBLE_SPACE in s_batch:
                        s_batch = s_batch.replace(DOUBLE_SPACE, SPACE)
                    if len(s_batch) >= batch_str_len:
                        break
                    s_ptr_batch_end += 1
                    if s_ptr_batch_end >= len(data_str):
                        s_ptr = 0
                        s_ptr_batch_end = batch_str_len
                batch_str_list.append(s_batch)
                s_ptr = s_ptr_batch_end

            loss_tensor = self.consume(*batch_str_list)
            loss_tensor.backward()
            optimiser.step()
            self.zero_grad()

            loss = loss_tensor.item()
            loss_list.append(loss)
            time_list.append(timer.time_taken())
            s_ptr += batch_size
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

        return time_list, loss_list

    def predict(self, prompt=None, num_chars=500, print_each_char=True):
        if prompt is None:
            prompt = ""

        if print_each_char:
            print(prompt, end="", flush=True)

        self._initialise_hidden_state(batch_size=1)
        if len(prompt) > 0:
            self.consume(prompt)

        char_pred_list = []
        for _ in range(num_chars):
            rnn_output = self._decoder_mlp.forward(self._hidden_state)
            exp_output = torch.exp(rnn_output)
            softmax_output = exp_output / torch.sum(exp_output)
            if self._cuda_device_id is not None:
                softmax_output = softmax_output.cpu()
            char_pred_id = self._rng.choice(
                len(self._char_list),
                p=softmax_output.detach().numpy().squeeze(),
            )
            char_pred = self._char_list[char_pred_id]
            char_one_hot = self._get_one_hot_char_tensor(char_pred)
            rnn_input = torch.cat([char_one_hot, self._hidden_state], axis=1)
            self._update_hidden_state(rnn_input)

            char_pred_list.append(char_pred)
            if print_each_char:
                print(char_pred, end="", flush=True)

        if print_each_char:
            print()

        return prompt + "".join(char_pred_list)

    def _get_char_id(self, c):
        return self._char_dict.get(c, self._default_char_id)

    def _get_one_hot_char_tensor(self, *chars):
        batch_size = len(chars)
        prev_batch_size, input_dim = self._one_hot_char_tensor.shape
        if prev_batch_size != batch_size:
            self._one_hot_char_tensor = torch.zeros(
                size=[batch_size, input_dim],
                dtype=torch.float32,
                device=self._one_hot_char_tensor.device,
            )
        else:
            self._one_hot_char_tensor *= 0.0
        for i, c in enumerate(chars):
            char_id = self._get_char_id(c)
            self._one_hot_char_tensor[i, char_id] = 1.0
        return self._one_hot_char_tensor
