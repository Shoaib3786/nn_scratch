
import torch
from src.layers import neuralNetwork
from src.activation import LeReLu_activation
from src.activation import Sigmoid_activation
from src.loss import Loss_CrossEntropy
from src.accuracy_metric import Accuracy
from src.data import DataGen
import argparse
import yaml
import pickle


def main():
  
  # Objects init.
  layer1 = neuralNetwork(num_neurons=l1_nodecount, number_features=l1_featurescount)
  layer2 = neuralNetwork(num_neurons=l2_nodecount, number_features=l2_featurescount)
  layer3 = neuralNetwork(num_neurons=l3_nodecount, number_features=l3_featurescount)
  LeKrelu_activation = LeReLu_activation()
  sigmoid_activation = Sigmoid_activation()
  loss_crossEntropy = Loss_CrossEntropy()
  accuracy = Accuracy()
  datagen = DataGen()
  X_train,y_train, X_test, y_test = datagen.dataSplit()
  batch_size=int(len(X_train)*batchsize_perc)
  

  # training loop
  for epochs in range(total_epoch):     # In each epochs (for each batches(fwd->loss & bwd->update weight) )
    print("\n**** epoch - {0} ****\n".format(epochs))

    for start_idx in range(0, len(X_train), batch_size):    # looping next batch in the whole dataset
      end_idx = start_idx + batch_size                      # creating slicing indexes

      # shiffling x_train in every epoch to not see repeatation of accuracy metric--better training
      perm = torch.randperm(len(X_train))
      X_train = X_train[perm]
      y_train = y_train[perm]

      """
      #############
      Forward prop --> (Z -> activation (for each layers till last layer) -> CrossEntropy loss (for last layer)
      #############
      """
      X_batch = X_train[start_idx:end_idx]        # X_train slicing into batch
      y_train_batch = y_train[start_idx:end_idx]  # y_train_batch slicing into batch

      layer1_output = layer1.forward(input_data=X_batch)
      layer1_relu_output = LeKrelu_activation.forward(node_z_value=layer1_output)

      layer2_output = layer2.forward(input_data=layer1_relu_output)
      layer2_relu_output = LeKrelu_activation.forward(node_z_value=layer2_output)

      # FINAL OUTPUT LAYER NEURON
      layer3_output = layer3.forward(input_data=layer2_relu_output)
      layer3_sigmoid_output = sigmoid_activation.forward(node_z_value=layer3_output)

      # LOSS
      loss = loss_crossEntropy.forward(y_pred=layer3_sigmoid_output, y_actual=y_train_batch)
      print("Loss: ", loss)

      # Accuracy Metrics
      accur = accuracy.calculate(y_pred=layer3_sigmoid_output, y_actual=y_train_batch)
      print("Accuracy: ", accur)


      """
      #############
      backward prop --> (loss -> activa -> Z -> input(prev_layer activation))
      #############
      d_l/d_w = (d_l/d_sigm) * (d_sigm/d_z) * (d_z/d_xw) * (d_xw/dw)
      d_l/d_b = (d_l/d_sigm) * (d_sigm/d_z) * (d_z/d_b)
      d_l/d_x = (d_l/d_sigm) * (d_sigm/d_z) * (d_z/d_xw) * (d_xw/dx)
      """
      ## Back-Calculation weight & bias for dw_3 and db_3:
      dl_dz_3 = loss_crossEntropy.backward(y_pred=layer3_sigmoid_output, y_actual=y_train_batch)
      dz3_dw_3 = layer2_relu_output                           # z3 = (w3 * relu_3) + b3
      
      dl_dw_3 = torch.matmul(dz3_dw_3.T, dl_dz_3)             # (2, batch) x (batch, 1) = (2,1)

      dz_db_3 = 1
      dl_db_3 = dl_dz_3 * dz_db_3                             # (1,1)
      dl_db_3 = dl_db_3.sum(dim=0)                            # (1,2)

      ## Back-Calculation weight & bias for dw2 and db_2:
      dz3_drelu_2 = layer3.weight                             # z3 = (w3 * relu_3) + b3
      dl_drelu_2 = torch.matmul(dl_dz_3, dz3_drelu_2.T)       # (batch,1) x (1,2) = (batch,2)
      drelu_dz_2 = LeKrelu_activation.backward(layer2_output) # (batch,2)
      dl_dz_2 = dl_drelu_2 * drelu_dz_2                       # (batch,2)
      dz2_dw_2 = layer1_relu_output                           # (2,batch)
      
      dl_dw_2 = torch.matmul(dz2_dw_2.T, dl_dz_2)             # (3, batch) x (batch, 2) = (3,2)

      dz_db_2 = 1
      dl_db_2 = dl_dz_2 * dz_db_2                             # (1,2)
      dl_db_2 = dl_db_2.sum(dim=0)                            # (1,2)

      ## Back-Calculation weight & bias for dw1 and db_1:
      dz2_drelu_1 = layer2.weight
      dl_drelu_1 = torch.matmul(dl_dz_2, dz2_drelu_1.T)         # (batch,2)x(2,3)=(batch,3)
      drelu_dz_1 = LeKrelu_activation.backward(layer1_output)   # (batch,3)
      dl_dz1 = dl_drelu_1 * drelu_dz_1                          # (batch,3)
      dz1_dw_1 = X_batch                                        # (batch,2)

      dl_dw_1 = torch.matmul(dz1_dw_1.T, dl_dz1)                # (2,batch)x(batch,3) = (2,3)

      dz_db_1 = 1
      dl_db_1 = dl_dz1 * dz_db_1                                # (1,3)
      dl_db_1 = dl_db_1.sum(dim=0)                              # (1,3)

      ## Update weights (in each layers)
      layer1.backward(dl_dw=dl_dw_1, dl_db=dl_db_1, rate= learn_rate)
      layer2.backward(dl_dw=dl_dw_2, dl_db=dl_db_2, rate= learn_rate)
      layer3.backward(dl_dw=dl_dw_3, dl_db=dl_db_3, rate= learn_rate)

      ## For visualization (Collecting (loss, accuracy & weight)##
      loss_history.append(loss.item())
      accuracy_history.append(accur.item())
      # flatten weights of one layer (say layer1) to track:
      weights_history.append(layer1.weight.detach().clone().numpy())


if __name__ == "__main__":

    # store logs --> used for Visualization
    loss_history = []
    accuracy_history = []
    weights_history = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str, default="config/param.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
 
    l1_nodecount=cfg["l1"]["nodes"]
    l1_featurescount=cfg['l1']['features']
    l2_nodecount=cfg["l2"]["nodes"]
    print(type(l2_nodecount))
    l2_featurescount = cfg["l2"]["features"]
    l3_nodecount=cfg["l3"]["nodes"]
    l3_featurescount = cfg["l3"]["features"]
    total_epoch = cfg["train"]["epochs"]
    learn_rate = cfg["train"]["lr"]
    batchsize_perc = cfg["train"]["batch_perc"]

    main()

    # After training loop
    logs = {
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "weights_history": weights_history,
    }

    with open("training_logs.pkl", "wb") as f:
      pickle.dump(logs, f)

