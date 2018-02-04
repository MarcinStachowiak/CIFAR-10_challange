import numpy as np
import matplotlib.pyplot as plt


def plot_images_per_class(dict, label_names):
    num_classes = len(dict)
    num_images=len(dict[0])
    f, axarr = plt.subplots(num_classes, num_images)
    for y_ax_num in range(0, num_classes):
        for x_ax_num in range(0, num_images):
            axes = axarr[y_ax_num][x_ax_num]
            axes.imshow(dict[y_ax_num][x_ax_num])
            axes.set_xticks([])
            axes.set_yticks([])
            if (x_ax_num == 0):
                l = axes.set_ylabel(label_names[y_ax_num], rotation='horizontal', fontsize=9,
                                    horizontalalignment='right', verticalalignment='center')
    plt.axis('off')
    plt.show()
