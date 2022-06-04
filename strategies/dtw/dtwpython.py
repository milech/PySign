# -----------------------------------------------------------
# Comparer of various implementations of Dynamic Time Warping (DTW) algorithm
# - DTW implementation from dtw-python library wrapped as a strategy following the strategy pattern
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from typing import List, Any, Union
from numpy.typing import NDArray
from dtw import dtw     # dtw-python package (T. Giorgino)
from dtwstrategy import DTWStrategy
from mydecorators import *


class DTWPythonStrategy(DTWStrategy):
    def __init__(self):
        self.__name = "DTW with dtw-python package"

    @plotresults
    @printcost2console
    @timer
    def do_algorithm(self, function_1: NDArray, function_2: NDArray) -> List[Union[NDArray, Any]]:
        # Find the best match with the canonical recursion formula
        alignment = dtw(function_1, function_2, keep_internals=True)
        return [self.__name, alignment.costMatrix, alignment.localCostMatrix, alignment.distance]
