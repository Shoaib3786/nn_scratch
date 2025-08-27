# nn_scratch

### How to experiment in this neural nets?
 step-1 Clone the repo
 step-2 Config/param.yaml change parameters according to your need for training
         if: want to add more layers, 
         then: 1) Add in the similar manner as that of previous layers in param.yaml
               2) And then update the changes in script/train.py file.

 step-3 run script/train.py --> for traning
 step-4 run script/visualize.py --> for graphical plots and video animation.


## My Learning in this project:
### Why accuracy and loss may show repetitive patterns?

If you notice repeating or stagnant patterns in accuracy and loss curves, the issue often lies in the dataset or the network’s training dynamics.

1. Dataset-related issues
- Improper dataset arrangement: Check if your train/test splits are correct and shuffled properly.
- Class imbalance: If one class dominates, the model may always predict that class. You can detect imbalance by checking the mean counts of unique labels.

2. Network not updating weights

- If your weights are not updating, your network may be “dead.”

---  Ways to verify:

    - Inspect gradients (∂L/∂W). If the gradients with respect to different layers’ weights don’t vary much (or stay close to zero), the network is not learning.

    - Monitor activations. If the activation output of one or more layers is nearly constant (close to zero or a fixed value) across batches, then the activations are “dead,” which leads to no weight updates during backpropagation.

3. Dead activation functions

- A common scenario is the “dying ReLU problem.” Since ReLU outputs zero for all negative inputs, neurons can get stuck outputting zero and never recover.

    - How to detect this:
    Print or plot the weights/activations of each layer during training. If many neurons are inactive (constant zero), you likely have dead units.

    - Possible fixes:
    Use variations of ReLU such as Leaky ReLU, Parametric ReLU (PReLU), or ELU, which allow small gradients for negative values.

4. Debugging tips

- Plot loss, accuracy, and weight values over epochs. If weights stay almost the same, learning isn’t happening.
- Print sample gradients per layer to ensure they’re flowing backward correctly.


## Graphs: 
1. HiddenLayer-2, Weights Evolution:
   --> weights were trained for a very small epoch for experimenting purpose only, but still quiet a similar variation in weights update.
<img width="707" height="467" alt="Screenshot 2025-08-27 at 4 45 18 AM" src="https://github.com/user-attachments/assets/37abe936-24e3-4c13-8fe9-94f665de2478" />

2 Accuracy and Loss graph:
<img width="993" height="492" alt="Screenshot 2025-08-27 at 4 45 41 AM" src="https://github.com/user-attachments/assets/486bd484-1fdf-499f-b849-853baf82c257" />
