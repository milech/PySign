# -----------------------------------------------------------
# Comparer of various implementations of Dynamic Time Warping (DTW) algorithm
# - original DTW implementation wrapped as a strategy following the strategy pattern
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from typing import List, Union, Any
import numpy as np
from numpy.typing import NDArray
from dtwstrategy import DTWStrategy
from mydecorators import *


# original DTW implementation as in Ident and Biopuap projects

class DTWOrigStrategy(DTWStrategy):
    def __init__(self):
        self.__name = "Original DTW"
        self.__costMatrix = None
        self.__distanceMatrix = None
        self.__total_cost = .0
        self.__path = np.empty((0, 2))

    @plotresults
    @printcost2console
    @timer
    def do_algorithm(self, function_1: NDArray, function_2: NDArray) -> List[Union[NDArray, Any]]:
        f_len = np.size(function_1)
        g_len = np.size(function_2)

        self.__costMatrix = np.full((f_len, g_len), np.inf)
        self.__costMatrix[0, 0] = .0

        self.__distanceMatrix = np.abs(function_1[:, None] - function_2[None, :])

        for i in range(1, f_len):
            for j in range(1, g_len):
                self.__costMatrix[i, j] = self.__distanceMatrix[i, j] + np.min([self.__costMatrix[i - 1, j - 1],
                                                                                self.__costMatrix[i - 1, j],
                                                                                self.__costMatrix[i, j - 1]])
        self.__traceback_path()

        return [self.__name, self.__costMatrix, self.__distanceMatrix, self.__total_cost]

    def __traceback_path(self) -> None:
        r = self.__costMatrix.shape[0] - 1
        c = self.__costMatrix.shape[1] - 1

        while r > 0 and c > 0:
            left_elem = self.__costMatrix[r, c - 1]
            diag_elem = self.__costMatrix[r - 1, c - 1]
            top_elem = self.__costMatrix[r - 1, c]

            # go left
            if left_elem < diag_elem and left_elem < top_elem:
                self.__total_cost += self.__distanceMatrix[r, c - 1]
                c = c - 1
            # go up
            elif top_elem < diag_elem and top_elem < left_elem:
                self.__total_cost += self.__distanceMatrix[r - 1, c]
                r = r - 1
            # go diagonal
            else:
                self.__total_cost += self.__distanceMatrix[r - 1, c - 1]
                r = r - 1
                c = c - 1

            self.__path = np.append(self.__path, [np.array([r, c])], axis=0)
