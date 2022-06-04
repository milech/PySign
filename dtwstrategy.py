# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
# - interface for DTW strategies following the strategy pattern
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from abc import ABC, abstractmethod
from numpy.typing import NDArray


class DTWStrategy(ABC):
    @abstractmethod
    def do_algorithm(self, function_1: NDArray, function_2: NDArray):
        pass
