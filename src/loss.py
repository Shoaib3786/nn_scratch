
import torch

class Loss_CrossEntropy:

  def forward(self, y_pred, y_actual):

    # Clip data to prevent division by log(0) =infinity error when y_pred=0, because for loss cal w'll be taking log of y_pred
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = torch.clamp(y_pred, 1e-7, 1 - 1e-7)

    # Binary cross-entropy formula
    sample_loss = -(y_actual * torch.log(y_pred_clipped) +
                    (1 - y_actual) * torch.log(1 - y_pred_clipped))
    batchAvg_loss = torch.mean(sample_loss)
    return batchAvg_loss


  """Derivative of CrossEnropy Loss directly with respect to pre-activation (Z) for faster calculation no need to perform chain rule explicitly(for sigmoid), already did on paper and included it"""
  def backward(self, y_pred, y_actual):     # d_l/d_z => dl/d_sigmoid * d_sigmoid/dz
    return y_pred - y_actual                # derivative of cross entropy function (included dl/d_sigmoid & d_sigmoid/dz)--> derived on paper manually


