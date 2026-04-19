from common import *
import numpy as np
from typing import *

def main(input_grid):
    grid = input_grid
    background = Color.BLACK
    connectivity = 4
    monochromatic = False
    from scipy.ndimage import label
    if connectivity == 4:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    connected_components = []
    if not monochromatic:
        labeled, n_objects = label(grid != background, structure)
        for i in range(n_objects):
            connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
            connected_components.append(connected_component)
    else:
        for color in set(grid.flatten()) - {background}:
            labeled, n_objects = label(grid == color, structure)
            for i in range(n_objects):
                connected_component = grid * (labeled == i + 1) + background * (labeled != i + 1)
                connected_components.append(connected_component)
    squares = connected_components
