
import torch

# RelU Activation function
class ReLu_activation:
  def forward(self, node_z_value):
    return torch.maximum(torch.tensor(0.0), node_z_value)

  def backward(self, node_z_value):
    return (node_z_value>0).float()

# LeakyReLu Activation function
class LeReLu_activation:   
    def forward(self, node_z_value):
        return torch.where(node_z_value > 0, node_z_value, 0.01*node_z_value)
    def backward(self, node_z_value):
        return torch.where(node_z_value > 0, torch.ones_like(node_z_value), torch.full_like(node_z_value, 0.01))

# Sigmoid Activation function
class Sigmoid_activation:

  def forward(self, node_z_value):
    return 1 / (1 + torch.exp(-node_z_value))

# SoftMax Activation function(#Last-layer)
class SoftMax_activation:   
  def forward(self, Lastlayer_output):
    exp_x = torch.exp(Lastlayer_output)

    # dim--> summation along dim=1(along feature for each sample), 
    # keepdim--> increase dim by 1 which is deleted due to summation.
    softmax = exp_x/torch.sum(exp_x, dim=1, keepdim=True)   
    return softmax
  
  