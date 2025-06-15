import matplotlib.pyplot as plt


def save_loss_curve(losses, xlabel, ylabel, title, save_path):
    plt.plot(losses)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()
