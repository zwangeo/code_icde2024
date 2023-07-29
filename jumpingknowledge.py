from typing import List, Optional
import torch
from torch import Tensor, nn, einsum
from torch.nn import LSTM, Linear, Parameter, ParameterList
import numpy as np


class JumpingKnowledge(torch.nn.Module):

    def __init__(self, mode: str, channels: Optional[int] = None,
                 num_layers: Optional[int] = None,
                 K: Optional[int] = None):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['concat', 'last', 'max', 'mean', 'att', 'gpr', 'lstm', 'node_adaptive']
        self.channels = channels
        self.att_scores = None
        self.gammas = None
        self.lstm, self.att = None, None
        self.s = None

        if self.mode == 'att':
            self.K = K
            self._initialize_att_scores()

        if self.mode == 'gpr':
            self.K = K
            self._initialize_gammas()

        if self.mode == 'lstm':
            assert self.channels is not None, 'channels cannot be None for lstm'
            assert num_layers is not None, 'num_layers cannot be None for lstm'
            self.lstm = LSTM(self.channels, (num_layers * self.channels) // 2, bidirectional=True, batch_first=True)
            self.att = Linear(2 * ((num_layers * self.channels) // 2), 1)

        if self.mode == 'node_adaptive':
            assert self.channels is not None, 'channels cannot be None for node_adaptive'
            self._initialize_s()

        self.reset_parameters()


    def _initialize_att_scores(self):
        ###################### random_init ##########################
        # TEMP = torch.rand(self.K+1)
        bound = np.sqrt(3/(self.K+1))
        TEMP = np.random.uniform(0, bound, self.K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.att_scores = Parameter(torch.tensor(TEMP, dtype=torch.float))


    def _initialize_gammas(self):
        ###################### random_init ##########################
        bound = np.sqrt(3/(self.K+1))
        TEMP = np.random.uniform(0, bound, self.K+1)
        TEMP = TEMP/np.sum(np.abs(TEMP))
        self.gammas = Parameter(torch.tensor(TEMP, dtype=torch.float))
        ###################### PPR_init ##########################
        # TEMP = self.alpha*(1-self.alpha)**np.arange(self.K+1)
        # TEMP[-1] = (1-self.alpha)**self.K


    def _initialize_s(self):
        self.s = Parameter(torch.Tensor(1, self.channels))
        nn.init.xavier_uniform_(self.s)


    def reset_parameters(self):
        # reset params for learnable candidates: gpr, lstm, node_adaptive
        if self.att_scores is not None:
            self._initialize_att_scores()
        if self.gammas is not None:
            self._initialize_gammas()
        if self.lstm is not None:
            self.lstm.reset_parameters()
        if self.att is not None:
            self.att.reset_parameters()
        if self.s is not None:
            self._initialize_s()


    def forward(self, xs: List[Tensor]) -> Tensor:
        if self.mode == 'last':
            return xs[-1]

        elif self.mode == 'concat':
            return torch.cat(xs, dim=-1)

        elif self.mode == 'max':
            return torch.stack(xs, dim=-1).max(dim=-1)[0]

        elif self.mode == 'mean':
            return torch.stack(xs, dim=-1).mean(dim=-1)

        elif self.mode == 'att':
            return einsum('K,KNd->Nd', torch.softmax(self.att_scores, dim=-1), torch.stack(xs))

        elif self.mode == 'gpr':
            # return einsum('i,ijk->jk', torch.tanh(self.gammas), torch.stack(xs))
            return einsum('K,KNd->Nd', torch.tanh(self.gammas), torch.stack(xs))

        elif self.mode == 'lstm':
            assert self.lstm is not None and self.att is not None
            x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
            alpha, _ = self.lstm(x)
            alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
            alpha = torch.softmax(alpha, dim=-1)
            return (x * alpha.unsqueeze(-1)).sum(dim=1)

        elif self.mode == 'node_adaptive':
            x = torch.stack(xs)  # [num_layers, num_nodes, num_channels]
            temp = einsum('KNd,Id->KN', x, self.s)
            scores = torch.tanh(temp)
            return einsum('KNd,KN->Nd', x, scores)

        else:
            raise NotImplementedError


    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mode})'