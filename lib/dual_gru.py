import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.jit as jit
from typing import List, Tuple
from torch import Tensor


class GRUCell(jit.ScriptModule):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih_1 = Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh_1 = Parameter(torch.randn(2 * hidden_size, hidden_size))
        
        self.weight_ih_2 = Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh_2 = Parameter(torch.randn(2 * hidden_size, hidden_size))
        
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

        self.zeros = torch.zeros(2 * hidden_size, input_size)
        
        self.weight_in_1 = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hn_1 = Parameter(torch.randn(hidden_size, hidden_size))
        
        self.weight_in_2 = Parameter(torch.randn(hidden_size, input_size))
        self.weight_hn_2 = Parameter(torch.randn(hidden_size, hidden_size))
        
        self.bias_in = Parameter(torch.randn(2 * hidden_size))
        self.bias_hn = Parameter(torch.randn(2 * hidden_size))

        self.zeros_n = torch.zeros(hidden_size, input_size)

    def pad(self, params_1, params_2, zeros):
        params_1 = torch.cat([params_1, zeros], 1)  # [2*hidden,2*input_size]
        params_2 = torch.cat([zeros, params_2], 1)  # [2*hidden,2*input_size]
        params = torch.cat([params_1,params_2],0)  # [4*hidden,2*input_size]
        return params

    @jit.script_method
    def forward(self, input, hx):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        weight_ih = self.pad(self.weight_ih_1,self.weight_ih_2, self.zeros.to(hx.device))
        weight_hh = self.pad(self.weight_hh_1,self.weight_hh_2, self.zeros.to(hx.device))
        
        weight_in = self.pad(self.weight_in_1, self.weight_in_2, self.zeros_n.to(hx.device)) # [2*hidden, 2*input_size]
        weight_hn = self.pad(self.weight_hn_1, self.weight_hn_2, self.zeros_n.to(hx.device))  
        
        gates = (torch.mm(input, weight_ih.t()) + self.bias_ih + \
                torch.mm(hx, weight_hh.t()) + self.bias_hh)
        resetgate_1, updategate_1, resetgate_2, updategate_2 = gates.chunk(4, 1)

        resetgate_1 = torch.sigmoid(resetgate_1)
        updategate_1 = torch.sigmoid(updategate_1)
        resetgate_2 = torch.sigmoid(resetgate_2)
        updategate_2 = torch.sigmoid(updategate_2)
        
        resetgate = torch.cat([resetgate_1, resetgate_2],1) # [batch, 2*hidden]
        updategate = torch.cat([updategate_1, updategate_2],1)
        
        cellgate = torch.mm(input, weight_in.t()) + self.bias_in + \
                resetgate * (torch.mm(hx, weight_hn.t()) + self.bias_hn)

        cellgate = torch.tanh(cellgate)  # [batch,2*hidden]

        hy = (1. - updategate) * cellgate + updategate * hx  # [batch,2*hidden]

        return hy, (hy)


class DualGRU(jit.ScriptModule):
    def __init__(self, cell, *cell_args):
        super(DualGRU, self).__init__()
        self.cell = cell(*cell_args)
    
    @jit.script_method
    def forward(self, input, state):
        # type: (Tensor, Tensor) -> Tuple[Tensor, Tensor]
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state

