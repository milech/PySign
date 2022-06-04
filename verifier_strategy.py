# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
#
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Any, Tuple


class VerifierStrategy(ABC):
    @abstractmethod
    def train(self, training_data: Any, classes: Tuple):
        pass

    @abstractmethod
    def verify(self, sample: Any, classes: Tuple):
        pass
