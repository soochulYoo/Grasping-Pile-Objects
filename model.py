import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
from torch.autograd import Variable
import time as timer
from tqdm import tqdm

import logging
logging.disable(logging.CRITICAL)
import scipy as sp
import scipy.sparse.linalg as spLA
import copy

import mjrl.samplers.core as trajectory_sampler
import mjrl.utils.process_samples as process_samples
from mjrl.utils.logger import DataLog
from mjrl.utils.cg_solve import cg_solve

from mjrl.algos.npg_cg import NPG

from typing import Tuple, Optional
from torch import Tensor
from torch import _VF
from torch.nn.modules import Module
from torch.nn import Parameter, init
import math

class CustomLSTMCell2(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, device:None) -> None:
        super(CustomLSTMCell2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size).cuda())
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size).cuda())
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size).cuda())
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size).cuda())
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)
    
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:

        # self.weight_hh = self.weight_hh.to(self.device)
        # self.weight_ih = self.weight_ih.to(self.device)
        # self.bias_hh   = self.bias_hh.to(self.device)
        # self.bias_ih   = self.bias_ih.to(self.device)

        print("===============")
        print(self.weight_hh.device)
        print(self.weight_ih.device)
        print(self.bias_hh.device)
        print(self.bias_ih.device)
        print("===============")
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx

        ret = _VF.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret


class CustomRNNCellBase(Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    # WARNING: bias_ih and bias_hh purposely not defined here.
    # See https://github.com/pytorch/pytorch/issues/39670

    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, device:None) -> None:
        super(CustomRNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.weight_ih = Parameter(torch.Tensor(num_chunks * hidden_size, input_size).to(self.device))
        self.weight_hh = Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size).to(self.device))
        if bias:
            self.bias_ih = Parameter(torch.Tensor(num_chunks * hidden_size).to(self.device))
            self.bias_hh = Parameter(torch.Tensor(num_chunks * hidden_size).to(self.device))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        print(self.weight_hh.device)
        print(self.weight_ih.device)
        print(self.bias_hh.device)
        print(self.bias_ih.device)
    def extra_repr(self) -> str:
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor) -> None:
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None:
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

class CustomLSTMCell(CustomRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(CustomLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4, device=device)
        self.device = device
        self.weight_hh.to(self.device)
        self.weight_ih.to(self.device)
        self.bias_hh.to(  self.device)
        self.bias_ih.to(  self.device)
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        assert input.dim() in (1, 2), \
            f"LSTMCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)
        else:
            hx = (hx[0].unsqueeze(0), hx[1].unsqueeze(0)) if not is_batched else hx
        print("===============")
        print(self.weight_hh.device)
        print(self.weight_ih.device)
        print(self.bias_hh.device)
        print(self.bias_ih.device)
        print("===============")
        ret = _VF.lstm_cell(
            input, hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

        if not is_batched:
            ret = (ret[0].squeeze(0), ret[1].squeeze(0))
        return ret

def pos_table(n, dim):
    def get_angle(x, h):
        return x / np.power(10000, 2 * (h // 2) / dim)

    def get_angle_vec(x):
        return [get_angle(x, j) for j in range(dim)]

    tab = np.array([get_angle_vec(i) for i in range(n)]).astype(float)
    tab[:, 0::2] = np.sin(tab[:, 0::2])
    tab[:, 1::2] = np.cos(tab[:, 1::2])
    return tab

class AttentionMatrix(nn.Module):
    def __init__(self, dim_in_q, dim_in_k, msg_dim, bias = True, scale = True):
        super(AttentionMatrix, self).__init__()
        self.proj_q = nn.Linear(in_features=dim_in_q, out_features=msg_dim, bias=bias)
        self.proj_k = nn.Linear(in_features=dim_in_k, out_features=msg_dim, bias=bias)
        if scale:
            self.msg_dim = msg_dim
        else:
            self.msg_dim = 1
    
    def forward(self, data_q, data_k):
        q = self.proj_q(data_q)
        k = self.proj_k(data_k)
        if data_q.ndim == data_k.ndim == 2:
            dot = torch.matmul(q, k.T)
        else:
            dot = torch.bmm(q, k.T)
        return torch.div(dot, np.sqrt(self.msg_dim))

class SelfAttentionMatrix(AttentionMatrix):
    def __init__(self, dim_in, msg_dim, bias = True, scale = True):
        super(SelfAttentionMatrix, self).__init__(
            dim_in_q=dim_in,
            dim_in_k=dim_in,
            msg_dim=msg_dim,
            bias=bias,
            scale=scale,
        )

class AttentionNeuronLayer(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, msg_dim, pos_em_dim, 
                    in_shift = None, in_scale = None,
                    out_shift = None, out_scale = None,
                    bias = True, scale = True):
        super(AttentionNeuronLayer, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.pos_embedding = torch.from_numpy(
            pos_table(self.hidden_dim, self.pos_em_dim)
        ).float() # hidden dim x pos_em_dim

        # self.device = device

        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size= 1 + self.act_dim, hidden_size=pos_em_dim)
        # self.lstm = self.lstm.cuda()

        self.attention = SelfAttentionMatrix(
            dim_in=pos_em_dim, msg_dim=self.msg_dim, bias=bias, scale=scale)
        
        # self.attention = self.attention.cuda()



    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        # self.in_shift  = torch.from_numpy(np.float32(in_shift)).to(self.device) if in_shift is not None else torch.zeros(self.obs_dim)
        # self.in_scale  = torch.from_numpy(np.float32(in_scale)).to(self.device) if in_scale is not None else torch.ones(self.obs_dim)
        # self.out_shift = torch.from_numpy(np.float32(out_shift)).to(self.device) if out_shift is not None else torch.zeros(self.act_dim)
        # self.out_scale = torch.from_numpy(np.float32(out_scale)).to(self.device) if out_scale is not None else torch.ones(self.act_dim)
    
    def forward(self, obs, prev_act):
        if isinstance(obs, np.ndarray):
            # x = torch.from_numpy(obs.copy()).float().unsqueeze(-1)
            x = torch.from_numpy(obs.copy()).float()
        else:
            # x = obs.unsqueeze(-1)
            x = obs
        if isinstance(prev_act, np.ndarray):
            prev_act = torch.from_numpy(prev_act.copy()).float()
        else:
            prev_act = prev_act
        obs_dim = x.shape[1]
        # obs_dim = self.obs_dim
        x = (x - self.in_shift)/(self.in_scale + 1e-8)

        output = None
        for idx in range(x.shape[0]):
            each_state = x[idx].unsqueeze(-1)
            each_action = prev_act[idx]
            state_action_pair = torch.cat([each_state, torch.vstack([each_action] * obs_dim)], dim = -1)
            if self.hx is None:
                self.hx = (
                torch.zeros(obs_dim, self.pos_em_dim).to(each_state.device),
                torch.zeros(obs_dim, self.pos_em_dim).to(each_state.device),
                )

            self.hx = self.lstm(state_action_pair, self.hx)

            w = torch.tanh(self.attention(
                data_q=self.pos_embedding.to(each_state.device), data_k=self.hx[0]))
            each_output = torch.matmul(w, each_state)
            if output == None:
                output = each_output.T
            else:
                output = torch.vstack((output, each_output.T))

        # x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)
        # if self.hx is None:
        #     self.hx = (
        #         torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
        #         torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
        #     )
        # self.hx = self.lstm(x_aug, self.hx)

        # w = torch.tanh(self.attention(
        #     data_q=self.pos_embedding.to(x.device), data_k=self.hx[0]))
        # output = torch.matmul(w, x)
        
        return torch.tanh(output)

    def reset(self):
        self.hx = None

class PermutationInvariantLayer(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, msg_dim, pos_em_dim, 
                    in_shift = None, in_scale = None,
                    out_shift = None, out_scale = None,
                    bias = True, scale = True, device = None):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.device = device
        self.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # store native scales that can be used for resets
        self.transformations = dict(in_shift=in_shift,
                           in_scale=in_scale,
                           out_shift=out_shift,
                           out_scale=out_scale
                          )
        # self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
        # self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
        # self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
        # self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
        self.in_shift  = torch.from_numpy(np.float32(in_shift)).to(self.device) if in_shift is not None else torch.zeros(self.obs_dim)
        self.in_scale  = torch.from_numpy(np.float32(in_scale)).to(self.device) if in_scale is not None else torch.ones(self.obs_dim)
        self.out_shift = torch.from_numpy(np.float32(out_shift)).to(self.device) if out_shift is not None else torch.zeros(self.act_dim)
        self.out_scale = torch.from_numpy(np.float32(out_scale)).to(self.device) if out_scale is not None else torch.ones(self.act_dim)
    
    def forward(self, x):
        return None



class FCNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 hidden_dim, msg_dim, pos_em_dim,
                 hidden_sizes=(64,64),
                 nonlinearity='tanh',   # either 'tanh' or 'relu'
                 in_shift = None,
                 in_scale = None,
                 out_shift = None,
                 out_scale = None
                 ):
        super(FCNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert type(hidden_sizes) == tuple
        self.layer_sizes = (obs_dim, ) + hidden_sizes + (act_dim, )

        # self.device = device

        self.attention_neuron = AttentionNeuronLayer(
            obs_dim =obs_dim,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim,
            in_shift = in_shift,
            in_scale = in_scale, 
            out_shift =out_shift,
            out_scale =out_scale
            # device = self.device
        )

        self.fc_layers = nn.ModuleList([nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]) \
                         for i in range(len(self.layer_sizes) -1)]) # 45 - 32 - 30
        self.nonlinearity = torch.relu if nonlinearity == 'relu' else torch.tanh
        self.out_shift = self.attention_neuron.out_shift
        self.out_scale = self.attention_neuron.out_scale
    
    #     self.set_transformations(in_shift, in_scale, out_shift, out_scale)

    # def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
    #     # store native scales that can be used for resets
    #     self.transformations = dict(in_shift=in_shift,
    #                        in_scale=in_scale,
    #                        out_shift=out_shift,
    #                        out_scale=out_scale
    #                       )
    #     # self.in_shift  = torch.from_numpy(np.float32(in_shift)) if in_shift is not None else torch.zeros(self.obs_dim)
    #     # self.in_scale  = torch.from_numpy(np.float32(in_scale)) if in_scale is not None else torch.ones(self.obs_dim)
    #     # self.out_shift = torch.from_numpy(np.float32(out_shift)) if out_shift is not None else torch.zeros(self.act_dim)
    #     # self.out_scale = torch.from_numpy(np.float32(out_scale)) if out_scale is not None else torch.ones(self.act_dim)
    #     self.in_shift  = torch.from_numpy(np.float32(in_shift)).to(self.device) if in_shift is not None else torch.zeros(self.obs_dim)
    #     self.in_scale  = torch.from_numpy(np.float32(in_scale)).to(self.device) if in_scale is not None else torch.ones(self.obs_dim)
    #     self.out_shift = torch.from_numpy(np.float32(out_shift)).to(self.device) if out_shift is not None else torch.zeros(self.act_dim)
    #     self.out_scale = torch.from_numpy(np.float32(out_scale)).to(self.device) if out_scale is not None else torch.ones(self.act_dim)

    def forward(self, obs, prev_act):
        out = self.attention_neuron(obs=obs, prev_act=prev_act)
        # if obs.is_cuda:
        #     out = obs.to(self.device)
        # else:
        #     out = obs
        # out = obs.to(self.device)
        # out = (out - self.in_shift)/(self.in_scale + 1e-8)


        for i in range(len(self.fc_layers)-1):
            # self.fc_layers[i] = self.fc_layers[i].to(self.device)
            out = self.fc_layers[i](out)
            out = self.nonlinearity(out)
        # self.fc_layers[-1] = self.fc_layers[-1].to(self.device)
        out = self.fc_layers[-1](out)
        out = out * self.out_scale + self.out_shift
        return out

class MLP:
    def __init__(self, env_spec,
                 hidden_sizes=(64,64),
                 min_log_std=-3,
                 init_log_std=0,
                 seed=None
                 ):
        """
        :param env_spec: specifications of the env (see utils/gym_env.py)
        :param hidden_sizes: network hidden layer sizes (currently 2 layers only)
        :param min_log_std: log_std is clamped at this value and can't go below
        :param init_log_std: initial log standard deviation
        :param seed: random seed
        """
        self.n = env_spec.observation_dim # number of states
        self.m = env_spec.action_dim  # number of actions
        self.min_log_std = min_log_std
        self.hidden_dim = self.n
        self.msg_dim = self.n
        self.pos_em_dim = self.m

        # Set seed
        # ------------------------
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Policy network
        # ------------------------
        # self.model = FCNetwork(self.n, self.m, hidden_sizes)
        self.model = FCNetwork(self.n, self.m, self.hidden_dim, self.msg_dim, self.pos_em_dim, hidden_sizes)
        # make weights small
        for param in list(self.model.parameters())[-2:]:  # only last layer
           param.data = 1e-2 * param.data
        self.log_std = Variable(torch.ones(self.m) * init_log_std, requires_grad=True)
        self.trainable_params = list(self.model.parameters()) + [self.log_std]

        # Old Policy network
        # ------------------------
        # self.old_model = FCNetwork(self.n, self.m, hidden_sizes)
        self.old_model = FCNetwork(self.n, self.m, self.hidden_dim, self.msg_dim, self.pos_em_dim, hidden_sizes)
        self.old_log_std = Variable(torch.ones(self.m) * init_log_std)
        self.old_params = list(self.old_model.parameters()) + [self.old_log_std]
        for idx, param in enumerate(self.old_params):
            param.data = self.trainable_params[idx].data.clone()

        # Easy access variables
        # -------------------------
        self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
        self.param_shapes = [p.data.numpy().shape for p in self.trainable_params]
        self.param_sizes = [p.data.numpy().size for p in self.trainable_params]
        # self.param_shapes = [p.data.cpu().numpy().shape for p in self.trainable_params]
        # self.param_sizes = [p.data.cpu().numpy().size for p in self.trainable_params]
        self.d = np.sum(self.param_sizes)  # total number of params

        # Placeholders
        # ------------------------
        self.obs_var = Variable(torch.randn(self.n), requires_grad=False)

    # Utility functions
    # ============================================
    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).data.numpy()
                                 for p in self.trainable_params])
        # params = np.concatenate([p.contiguous().view(-1).data.cpu().numpy()
        #                          for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params, set_new=True, set_old=True):
        if set_new:
            current_idx = 0
            for idx, param in enumerate(self.trainable_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.trainable_params[-1].data = \
                torch.clamp(self.trainable_params[-1], self.min_log_std).data
            # update log_std_val for sampling
            self.log_std_val = np.float64(self.log_std.data.numpy().ravel())
            # self.log_std_val = np.float64(self.log_std.data.cpu().numpy().ravel())
        if set_old:
            current_idx = 0
            for idx, param in enumerate(self.old_params):
                vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
                vals = vals.reshape(self.param_shapes[idx])
                param.data = torch.from_numpy(vals).float()
                current_idx += self.param_sizes[idx]
            # clip std at minimum value
            self.old_params[-1].data = \
                torch.clamp(self.old_params[-1], self.min_log_std).data

    # Main functions
    # ============================================
    def get_action(self, observation):
        o = np.float32(observation.reshape(1, -1))
        self.obs_var.data = torch.from_numpy(o)
        mean = self.model(self.obs_var).data.numpy().ravel()
        # mean = self.model(self.obs_var).data.cpu().numpy().ravel()
        noise = np.exp(self.log_std_val) * np.random.randn(self.m)
        action = mean + noise
        return [action, {'mean': mean, 'log_std': self.log_std_val, 'evaluation': mean}]

    def mean_LL(self, observations, actions, model=None, log_std=None):
        model = self.model if model is None else model
        log_std = self.log_std if log_std is None else log_std
        if type(observations) is not torch.Tensor:
            obs_var = Variable(torch.from_numpy(observations).float(), requires_grad=False)
        else:
            obs_var = observations
        if type(actions) is not torch.Tensor:
            act_var = Variable(torch.from_numpy(actions).float(), requires_grad=False)
        else:
            act_var = actions
        mean = model(obs_var)
        zs = (act_var - mean) / torch.exp(log_std)
        LL = - 0.5 * torch.sum(zs ** 2, dim=1) + \
             - torch.sum(log_std) + \
             - 0.5 * self.m * np.log(2 * np.pi)
        return mean, LL

    def log_likelihood(self, observations, actions, model=None, log_std=None):
        mean, LL = self.mean_LL(observations, actions, model, log_std)
        return LL.data.numpy()
        # return LL.data.cpu().numpy()

    def old_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.old_model, self.old_log_std)
        return [LL, mean, self.old_log_std]

    def new_dist_info(self, observations, actions):
        mean, LL = self.mean_LL(observations, actions, self.model, self.log_std)
        return [LL, mean, self.log_std]

    def likelihood_ratio(self, new_dist_info, old_dist_info):
        LL_old = old_dist_info[0]
        LL_new = new_dist_info[0]
        LR = torch.exp(LL_new - LL_old)
        return LR

    def mean_kl(self, new_dist_info, old_dist_info):
        old_log_std = old_dist_info[2]
        new_log_std = new_dist_info[2]
        old_std = torch.exp(old_log_std)
        new_std = torch.exp(new_log_std)
        old_mean = old_dist_info[1]
        new_mean = new_dist_info[1]
        Nr = (old_mean - new_mean) ** 2 + old_std ** 2 - new_std ** 2
        Dr = 2 * new_std ** 2 + 1e-8
        sample_kl = torch.sum(Nr / Dr + new_log_std - old_log_std, dim=1)
        return torch.mean(sample_kl) 


class BehaviorCloning():
    def __init__(self, expert_paths,
                 policy,
                 epochs = 5,
                 batch_size = 64,
                 lr = 1e-3,
                 optimizer = None,
                 loss_type = 'MSE',  # can be 'MLE' or 'MSE'
                 save_logs = True,
                 set_transforms = False,
                 
                 **kwargs,
                 ):

        self.policy = policy
        self.expert_paths = expert_paths
        self.epochs = epochs
        self.mb_size = batch_size
        self.logger = DataLog()
        self.loss_type = loss_type
        self.save_logs = save_logs

        # self.device = device

        if set_transforms:
            in_shift, in_scale, out_shift, out_scale = self.compute_transformations()
            self.set_transformations(in_shift, in_scale, out_shift, out_scale)
            self.set_variance_with_data(out_scale)

        # construct optimizer
        self.optimizer = torch.optim.Adam(self.policy.trainable_params, lr=lr) if optimizer is None else optimizer

        # Loss criterion if required
        if loss_type == 'MSE':
            self.loss_criterion = torch.nn.MSELoss()

        # make logger
        if self.save_logs:
            self.logger = DataLog()

    def compute_transformations(self):
        # get transformations
        if self.expert_paths == [] or self.expert_paths is None:
            in_shift, in_scale, out_shift, out_scale = None, None, None, None
        else:
            observations = np.concatenate([path["observations"] for path in self.expert_paths])
            actions = np.concatenate([path["actions"] for path in self.expert_paths])
            in_shift, in_scale = np.mean(observations, axis=0), np.std(observations, axis=0)
            out_shift, out_scale = np.mean(actions, axis=0), np.std(actions, axis=0)
        return in_shift, in_scale, out_shift, out_scale

    def set_transformations(self, in_shift=None, in_scale=None, out_shift=None, out_scale=None):
        # set scalings in the target policy
        # self.policy.model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        # self.policy.old_model.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.model.attention_neuron.set_transformations(in_shift, in_scale, out_shift, out_scale)
        self.policy.old_model.attention_neuron.set_transformations(in_shift, in_scale, out_shift, out_scale)

    def set_variance_with_data(self, out_scale):
        # set the variance of gaussian policy based on out_scale
        params = self.policy.get_param_values()
        params[-self.policy.m:] = np.log(out_scale + 1e-12)
        self.policy.set_param_values(params)

    def loss(self, data, idx=None):
        if self.loss_type == 'MLE':
            return self.mle_loss(data, idx)
        elif self.loss_type == 'MSE':
            return self.mse_loss(data, idx)
        else:
            print("Please use valid loss type")
            return None

    def mle_loss(self, data, idx):
        # use indices if provided (e.g. for mini-batching)
        # otherwise, use all the data
        idx = range(data['observations'].shape[0]) if idx is None else idx
        if type(data['observations']) == torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act = data['expert_actions'][idx]
        LL, mu, log_std = self.policy.new_dist_info(obs, act)
        # minimize negative log likelihood
        return -torch.mean(LL)

    def mse_loss(self, data, idx=None):
        idx = range(data['observations'].shape[0]) if idx is None else idx # 19884 x 45
        if type(data['observations']) is torch.Tensor:
            idx = torch.LongTensor(idx)
        obs = data['observations'][idx]
        act_expert = data['expert_actions'][idx] # 19884 x 30
        if type(data['observations']) is not torch.Tensor:
            obs = Variable(torch.from_numpy(obs).float(), requires_grad=False)
            act_expert = Variable(torch.from_numpy(act_expert).float(), requires_grad=False)
            # obs = Variable(torch.from_numpy(obs).float().to(self.device), requires_grad=False)
            # act_expert = Variable(torch.from_numpy(act_expert).float().to(self.device), requires_grad=False)
        # else:
        #     obs = obs.to(self.device)
        #     act_expert = act_expert.to(self.device)
        # act_pi = self.policy.model(obs)
        act_pi = self.policy.model(obs, act_expert) # 19884 x 30
        return self.loss_criterion(act_pi, act_expert.detach())

    def fit(self, data, suppress_fit_tqdm=False, **kwargs):
        # data is a dict
        # keys should have "observations" and "expert_actions"
        validate_keys = all([k in data.keys() for k in ["observations", "expert_actions"]])
        assert validate_keys is True
        ts = timer.time()
        num_samples = data["observations"].shape[0]

        # log stats before
        if self.save_logs:
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            # loss_val = self.loss(data, idx=range(num_samples)).data.cpu().numpy().ravel()[0]
            self.logger.log_kv('loss_before', loss_val)
        
        # train loop
        for ep in config_tqdm(range(self.epochs), suppress_fit_tqdm):
            for mb in range(int(num_samples / self.mb_size)):

                self.policy.model.attention_neuron.reset()

                rand_idx = np.random.choice(num_samples, size=self.mb_size) # 32 x 1
                self.optimizer.zero_grad()
                loss = self.loss(data, idx=rand_idx) # return nn.MSELoss(act_bc, act_expert)
                loss.backward()
                self.optimizer.step()
        params_after_opt = self.policy.get_param_values()
        self.policy.set_param_values(params_after_opt, set_new=True, set_old=True)

        # log stats after
        if self.save_logs:
            self.logger.log_kv('epoch', self.epochs)
            loss_val = self.loss(data, idx=range(num_samples)).data.numpy().ravel()[0]
            # loss_val = self.loss(data, idx=range(num_samples)).data.cpu().numpy().ravel()[0]
            self.logger.log_kv('loss_after', loss_val)
            self.logger.log_kv('time', (timer.time()-ts))

    def train(self, **kwargs):
        observations = np.concatenate([path["observations"] for path in self.expert_paths])
        expert_actions = np.concatenate([path["actions"] for path in self.expert_paths])
        data = dict(observations=observations, expert_actions=expert_actions)
        self.fit(data, **kwargs)


def config_tqdm(range_inp, suppress_tqdm=False):
    if suppress_tqdm:
        return range_inp
    else:
        return tqdm(range_inp)

class DAPG(NPG):
    def __init__(self, env, policy, baseline,
                 demo_paths=None,
                 normalized_step_size=0.01,
                 FIM_invert_args={'iters': 10, 'damping': 1e-4},
                 hvp_sample_frac=1.0,
                 seed=123,
                 save_logs=False,
                 kl_dist=None,
                 lam_0=1.0,  # demo coef
                 lam_1=0.95, # decay coef
                 **kwargs,
                 ):

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.kl_dist = kl_dist if kl_dist is not None else 0.5*normalized_step_size
        self.seed = seed
        self.save_logs = save_logs
        self.FIM_invert_args = FIM_invert_args
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.demo_paths = demo_paths
        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.iter_count = 0.0
        if save_logs: self.logger = DataLog()

    def train_from_paths(self, paths):

        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        if self.demo_paths is not None and self.lam_0 > 0.0:
            demo_obs = np.concatenate([path["observations"] for path in self.demo_paths])
            demo_act = np.concatenate([path["actions"] for path in self.demo_paths])
            demo_adv = self.lam_0 * (self.lam_1 ** self.iter_count) * np.ones(demo_obs.shape[0])
            self.iter_count += 1
            # concatenate all
            all_obs = np.concatenate([observations, demo_obs])
            all_act = np.concatenate([actions, demo_act])
            all_adv = 1e-2*np.concatenate([advantages/(np.std(advantages) + 1e-8), demo_adv])
        else:
            all_obs = observations
            all_act = actions
            all_adv = advantages

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        self.running_score = mean_return if self.running_score is None else \
                             0.9*self.running_score + 0.1*mean_return  # approx avg of last 10 iters
        if self.save_logs: self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # DAPG
        ts = timer.time()
        sample_coef = all_adv.shape[0]/advantages.shape[0]
        dapg_grad = sample_coef*self.flat_vpg(all_obs, all_act, all_adv)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions],
                                  regu_coef=self.FIM_invert_args['damping'])
        npg_grad = cg_solve(hvp, dapg_grad, x_0=dapg_grad.copy(),
                            cg_iters=self.FIM_invert_args['iters'])
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        n_step_size = 2.0*self.kl_dist
        alpha = np.sqrt(np.abs(n_step_size / (np.dot(dapg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params, set_new=True, set_old=False)
        surr_after = self.CPI_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        kl_dist = self.kl_old_new(observations, actions).data.numpy().ravel()[0]
        self.policy.set_param_values(new_params, set_new=True, set_old=True)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)
            try:
                self.env.env.env.evaluate_success(paths, self.logger)
            except:
                # nested logic for backwards compatibility. TODO: clean this up.
                try:
                    success_rate = self.env.env.env.evaluate_success(paths)
                    self.logger.log_kv('success_rate', success_rate)
                except:
                    pass
        return 

