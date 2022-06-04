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


class FixedThrVerifierStrategy(VerifierStrategy):
    def train(self):
        pass

    def verify(self, sample: Any, classes: Tuple) -> Any:
        return 0
