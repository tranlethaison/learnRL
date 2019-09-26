import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs
        self.make_plots()

    def make_plots(self):
        self.fig, self.axes = plt.subplots(2, 1, sharex=True)

        # Losses
        self.loss_data = [None] * self.n_epochs
        self.loss_id = 0
        self.axes[0].set_ylabel("Loss")
        self.loss, = self.axes[0].plot(self.loss_data)

        # Accuracy
        self.accu_data = [None] * self.n_epochs
        self.accu_id = 0
        self.axes[1].set_ylabel("Accuracy")
        self.accu, = self.axes[1].plot(self.accu_data)

        self.axes[-1].set_xlim(0, self.n_epochs - 1)
        self.axes[-1].set_xlabel("Epoch")

        for ax in self.axes:
            ax.grid(True)

    def update_loss(self, loss):
        self.loss_data[self.loss_id] = loss
        self.loss_id += 1
        self.loss.set_ydata(self.loss_data)

    def update_accu(self, accuracy):
        self.accu_data[self.accu_id] = accuracy
        self.accu_id += 1
        self.accu.set_ydata(self.accu_data)
