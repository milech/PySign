# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
# - DTW context following the strategy pattern
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from numpy.typing import NDArray
from dtwstrategy import DTWStrategy


class DTWContext:
    def __init__(self, dtw_strategy: DTWStrategy) -> None:
        self.__name = None
        self.__costMatrix = None
        self.__distanceMatrix = None
        self.__total_cost = None
        self.__path = None
        self.__dtw_strategy = dtw_strategy

    @property
    def gamma(self) -> NDArray:
        return self.__costMatrix

    @property
    def distance(self) -> NDArray:
        return self.__distanceMatrix

    @property
    def total_cost(self) -> float:
        return self.__total_cost

    @property
    def path(self) -> NDArray:
        return self.__path

    @property
    def dtw_strategy(self) -> DTWStrategy:
        return self.__dtw_strategy

    @dtw_strategy.setter
    def dtw_strategy(self, dtw_strategy: DTWStrategy) -> None:
        self.__dtw_strategy = dtw_strategy

    def do_algorithm(self, function_1: NDArray, function_2: NDArray) -> None:
        [self.__name,
         self.__costMatrix,
         self.__distanceMatrix,
         self.__total_cost] = self.__dtw_strategy.do_algorithm(function_1, function_2)
