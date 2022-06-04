# -----------------------------------------------------------
# Comparer of various implementations of Dynamic Time Warping (DTW) algorithm
# - DTW implementation by PG WETI AI Tech students:
# Krzysztof Walentukiewicz, Aleksandra Gałka, Justyna Jelińska, Albert Masiak
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------
from typing import List, Union, Any
from numpy.typing import NDArray
from dtwstrategy import DTWStrategy
from mydecorators import timer


class DTWStudentStrategy(DTWStrategy):
    def __init__(self):
        self.__name = "DTW implemented by students"

    @timer
    def do_algorithm(self, function_1: NDArray, function_2: NDArray) -> List[Union[NDArray, Any]]:
        # students' code to be placed here
        return [self.__name, None, None, None]
