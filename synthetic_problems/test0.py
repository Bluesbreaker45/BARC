from common import *
import numpy as np
from typing import *

def main(input_grid):
    squares = find_connected_components(input_grid, background=Color.BLACK, monochromatic=False)