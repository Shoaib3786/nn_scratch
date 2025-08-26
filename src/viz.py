import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from scripts.train import weights_history, accuracy_history, loss_history


# --------- VISUALIZATION for Weights of Layer-1 --------- #

# weight history into numpy array
weights_history = np.array(weights_history)  # shape = (epochs, in_features, out_features)
print("weights_history shape:", weights_history.shape)
fig = plt.figure(figsize=(8,5))
for i in range(weights_history.shape[1]):   # iterate over all weights
    plt.plot(range(weights_history.shape[0]), weights_history[:,i,0], label=f'w{i+1}')

plt.xlabel("Step")
plt.ylabel("Weight Value")
plt.title("Evolution of Layer2 Weights")
plt.legend()
plt.grid(True)
plt.show()


# --------- ANIMATION for LOSS & ACCURACY --------- #

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

# init plots
loss_line, = ax1.plot([], [], lw=2, label="Loss")
ax1.set_xlim(0, len(loss_history))
ax1.set_ylim(min(loss_history)-0.1, max(loss_history)+0.1)
ax1.set_title("Loss Convergence")
ax1.set_xlabel("Batch")
ax1.set_ylabel("Loss")

# init plots
accuracy_line, = ax2.plot([], [], lw=2, label="Accuracy")
ax2.set_xlim(0, len(accuracy_history))
ax2.set_ylim(min(accuracy_history)-0.1, max(accuracy_history)+0.1)
ax2.set_title("Accuracy")
ax2.set_xlabel("Batch")
ax2.set_ylabel("Loss")

def init():
    loss_line.set_data([], [])
    accuracy_line.set_data([], [])
    return loss_line, accuracy_line

def update(frame):
    # loss curve
    loss_line.set_data(range(frame), loss_history[:frame])

    # accuracy curve
    accuracy_line.set_data(range(frame), accuracy_history[:frame])
    return loss_line, accuracy_line

ani = animation.FuncAnimation(fig, update, frames=len(loss_history),
                              init_func=init, blit=True, interval=100)

plt.tight_layout()
HTML(ani.to_jshtml())

