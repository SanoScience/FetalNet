import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])

                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def double_conv(input_channels, output_channels):
    """
    Args:
        input_channels:
        output_channels:
    Returns:
    """
    conv = nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_features=output_channels),
        nn.ReLU(inplace=True),
        nn.Dropout2d(p=0.2)
    )
    return conv


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')

    return src


class YNet(nn.Module):
    """
    UNet class.
    """

    def __init__(self, input_channels, output_channels, n_class):
        """
        Args:
            input_channels:
            output_channels:
            n_class:
        """
        super(YNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv1 = double_conv(input_channels, output_channels)
        self.down_conv2 = double_conv(output_channels, 2 * output_channels)
        self.down_conv3 = double_conv(2 * output_channels, 4 * output_channels)
        self.down_conv4 = double_conv(4 * output_channels, 8 * output_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=8 * output_channels, out_features=4)
        )

        self.bootleneck = double_conv(8 * output_channels, 16 * output_channels)

        self.up_transpose4 = nn.ConvTranspose2d(16 * output_channels, 8 * output_channels, kernel_size=2, stride=2)
        self.Att4 = Attention_block(F_g=8 * output_channels, F_l=8 * output_channels, F_int=4 * output_channels)
        self.up_conv4 = double_conv(16 * output_channels, 8 * output_channels)

        self.up_transpose3 = nn.ConvTranspose2d(8 * output_channels, 4 * output_channels, kernel_size=2, stride=2)
        self.Att3 = Attention_block(F_g=4 * output_channels, F_l=4 * output_channels, F_int=2 * output_channels)
        self.up_conv3 = double_conv(8 * output_channels, 4 * output_channels)

        self.up_transpose2 = nn.ConvTranspose2d(4 * output_channels, 2 * output_channels, kernel_size=2, stride=2)
        self.Att2 = Attention_block(F_g=2 * output_channels, F_l=2 * output_channels, F_int=1 * output_channels)
        self.up_conv2 = double_conv(4 * output_channels, 2 * output_channels)

        self.up_transpose1 = nn.ConvTranspose2d(2 * output_channels, output_channels, kernel_size=2, stride=2)
        self.Att1 = Attention_block(F_g=1 * output_channels, F_l=1 * output_channels, F_int=output_channels // 2)
        self.up_conv1 = double_conv(2 * output_channels, output_channels)
        self.out_layer = nn.Conv2d(4, out_channels=n_class, kernel_size=3, padding=1)
        self.conv_lstm = ConvLSTM(input_size=(14, 14),
                                  input_dim=512,
                                  hidden_dim=[512, 512],
                                  kernel_size=(3, 3),
                                  num_layers=2,
                                  batch_first=False,
                                  bias=True,
                                  return_all_layers=False)
        self.side1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Args:
            image:
        Returns:
        """
        # Encoder part
        x = torch.unbind(x, dim=1)
        data = []
        for item in x:
            x1 = self.down_conv1(item)  # [1, 64, 224, 224]
            x2 = self.down_conv2(self.max_pool_2x2(x1))  # # [1, 128, 112, 112]
            x3 = self.down_conv3(self.max_pool_2x2(x2))  # [1, 256, 56, 56]
            x4 = self.down_conv4(self.max_pool_2x2(x3))  # [1, 512, 28, 28]
            features = self.max_pool_2x2(x4)  # [1, 512, 14, 14]
            data.append(features.unsqueeze(0))
        data = torch.cat(data, dim=0)  # [1, 1, 512, 14, 14]
        lstm, _ = self.conv_lstm(data)
        test = lstm[0][-1, :, :, :, :]  # [1, 512, 14, 14]
        center = self.bootleneck(test)  # [1, 1024, 14, 14]

        # Classification
        avg_pool = self.avgpool(test)
        flatten = torch.flatten(avg_pool, 1)
        fc = self.classifier(flatten)

        # Decoder part
        up_transpose4 = self.up_transpose4(center)
        att4 = self.Att4(g=up_transpose4, x=x4)
        up_conv4 = self.up_conv4(torch.cat([x4, att4], dim=1))  # [1, 512, 28, 28]

        up_transpose3 = self.up_transpose3(up_conv4)
        att3 = self.Att3(g=up_transpose3, x=x3)
        up_conv3 = self.up_conv3(torch.cat([x3, att3], dim=1))  # [1, 256, 56, 56]

        up_transpose2 = self.up_transpose2(up_conv3)
        att2 = self.Att2(g=up_transpose2, x=x2)
        up_conv2 = self.up_conv2(torch.cat([x2, att2], dim=1))  # [1, 128, 112, 112]

        up_transpose1 = self.up_transpose1(up_conv2)
        att1 = self.Att1(g=up_transpose1, x=x1)
        up_conv1 = self.up_conv1(torch.cat([x1, att1], dim=1))  # [1, 64, 224, 224]

        # Side output
        s_1 = self.side1(up_conv1)  # [1, 1, 224, 224]
        s_2 = self.side2(up_conv2)
        s_2 = _upsample_like(s_2, s_1)
        s_3 = self.side3(up_conv3)
        s_3 = _upsample_like(s_3, s_1)
        s_4 = self.side4(up_conv4)
        s_4 = _upsample_like(s_4, s_1)
        out = self.out_layer(torch.cat((s_1, s_2, s_3, s_4), 1))  # [1, 1, 224, 224]
        out = torch.sigmoid(out)

        return fc, out
