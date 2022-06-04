# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
#
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from typing import Any, Tuple
from verifier_strategy import VerifierStrategy


class VerifierContext:
    def __init__(self, verifier_strategy: VerifierStrategy) -> None:
        self.__verifier_strategy = verifier_strategy

    @property
    def verifier_strategy(self) -> VerifierStrategy:
        return self.__verifier_strategy

    @verifier_strategy.setter
    def verifier_strategy(self, verifier_strategy: VerifierStrategy) -> None:
        self.__verifier_strategy = verifier_strategy

    def train(self, training_data: Any, classes: Tuple):
        self.__verifier_strategy.train(training_data, classes)

    def verify(self, sample: Any, classes: Any) -> float:
        return self.__verifier_strategy.verify(sample, classes)
