from typing import List, Union, Any
from numpy.typing import NDArray
from fastdtw import fastdtw
from dtwstrategy import DTWStrategy
from mydecorators import timer
from mydecorators import printcost2console


class FastDTWStrategy(DTWStrategy):
    def __init__(self):
        self.__name = "DTW with fastdtw package"

    @printcost2console
    @timer
    def do_algorithm(self, function_1: NDArray, function_2: NDArray) -> List[Union[NDArray, Any]]:
        total_cost, _ = fastdtw(function_1, function_2)
        return [self.__name, None, None, total_cost]
