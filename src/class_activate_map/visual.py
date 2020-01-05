import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc', size=18)


def cam_visualization(cam_arr, text_fragment_lst):
    fig, ax = plt.subplots(figsize=(10, 24))
    im = ax.imshow(cam_arr, cmap="YlGn")
    # We want to show x ticks...
    ax.set_xticks(np.arange(len(text_fragment_lst)))
    ax.set_yticks([])
    # ... and label them with the respective list entries
    ax.set_xticklabels(text_fragment_lst, fontproperties=font)
    #
    for i in range(len(text_fragment_lst)):
        text = ax.text(i, 0, '{}%'.format(int(100 * cam_arr[0, i])),
                       ha="center", va="center")
    fig.tight_layout()
    plt.show()
