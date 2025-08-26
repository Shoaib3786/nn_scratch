
import torch
import math

"""
weight matrix = numb(inp_features/or previous layer node count) x numb(current_neurons_count)
input matrix = numb(batchsize) x numb(inp_features)
"""

class neuralNetwork:

  def __init__(self, num_neurons, number_features):
    self.num_neurons = num_neurons
    self.bias= torch.ones(size=(1, self.num_neurons), dtype=torch.float32)
    self.weight = torch.randn((number_features, num_neurons)) * (1.0 / math.sqrt(number_features))

  def forward(self, input_data):    # forward proporgation
    self.X = input_data
    self.output = torch.matmul(self.X, self.weight) + self.bias
    return self.output

  def backward(self, dl_dw, dl_db, rate):   # backward proporgation(param update)
    self.weight -= rate * dl_dw
    self.bias -= rate * dl_db