import torch.nn as nn
import torch

from model.HDDRN.HighFreqNetwork import HighFreqNetwork


class HDDRN(nn.Module):
    def __init__(self):
        super(HDDRN, self).__init__()

        self.group1Net = HighFreqNetwork(2, 2)
        self.group2Net = HighFreqNetwork(3, 3)
        self.group3Net = HighFreqNetwork(3, 3)

    def forward(self, x_high_freq):
        x_group1, x_group2, x_group3 = (x_high_freq[:, 0:2, :, :],
                                        x_high_freq[:, 2:5, :, :],
                                        x_high_freq[:, 5:8, :, :])

        group1_output = self.group1Net(x_group1)
        group2_output = self.group2Net(x_group2)
        group3_output = self.group3Net(x_group3)

        coe_output = torch.cat((group1_output, group2_output, group3_output), dim=1)

        return coe_output


class DWTHDDRN(nn.Module):
    """Refine the three Haar high-frequency subbands independently."""

    def __init__(self):
        super(DWTHDDRN, self).__init__()
        self.hl_net = HighFreqNetwork(1, 1)
        self.lh_net = HighFreqNetwork(1, 1)
        self.hh_net = HighFreqNetwork(1, 1)

    def forward(self, x_high_freq):
        if x_high_freq.shape[1] != 3:
            raise ValueError(
                f'DWTHDDRN expects three high-frequency subbands, got {x_high_freq.shape[1]}.'
            )

        outputs = (
            self.hl_net(x_high_freq[:, 0:1, :, :]),
            self.lh_net(x_high_freq[:, 1:2, :, :]),
            self.hh_net(x_high_freq[:, 2:3, :, :]),
        )
        return torch.cat(outputs, dim=1)
