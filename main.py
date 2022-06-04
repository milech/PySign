# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
# - entry point
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

from typing import List, Union, Tuple, Any
from multiprocessing import Process
import numpy as np
from numpy.typing import NDArray
from dtwcontext import DTWContext
from dtwstrategy import DTWStrategy
from strategies.dtw.origdtw import DTWOrigStrategy
from strategies.dtw.dtwpython import DTWPythonStrategy
# from strategies.dtw.fastdtwstrat import FastDTWStrategy
from verifier_context import VerifierContext
from verifier_strategy import VerifierStrategy
from strategies.verifiers.cnn_verifier import CNNVerifierStrategy
# from strategies.verifiers.fixed_thr_verifier import FixedThrVerifier
from mydecorators import timer4main


def run_strategy(*args: Union[DTWStrategy, NDArray, NDArray]):
    dtw_strategy, function_1, function_2 = args[0], args[1], args[2]
    dtw_context = DTWContext(dtw_strategy)
    dtw_context.do_algorithm(function_1, function_2)


def run_strategies(dtw_strategies: List, function_1: NDArray, function_2: NDArray) -> None:
    processes = []
    for dtw_strategy in dtw_strategies:
        p = Process(target=run_strategy, args=(dtw_strategy, function_1, function_2))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()


# TODO: refactor using template method pattern or so
def run_verifier_strategy(verifier_strategy: VerifierStrategy, training_data: Any, samples: NDArray, classes: Tuple):
    verifier_context = VerifierContext(verifier_strategy)
    verifier_context.train(training_data, classes)
    verifier_context.verify(samples, classes)


# TODO: parallelize and refactor using template method pattern or so
def run_verifier_strategies(verifier_strategies: List, training_data: Any, samples: NDArray, classes: Tuple) -> None:
    for verify_strategy in verifier_strategies:
        run_verifier_strategy(verify_strategy, training_data, samples, classes)


# @timer4main
def main():
    if __name__ == '__main__':
        sample_size = 1000
        # use sin and its deviation to test DTW
        l_space = np.linspace(-np.pi, np.pi, sample_size)
        function_1 = np.sin(l_space)
        function_2 = (np.random.uniform(0.1, 4)) * np.sin(l_space + np.random.rand(1) * np.pi)

        dtw_strategies = [DTWOrigStrategy(), DTWPythonStrategy()]
        run_strategies(dtw_strategies, function_1, function_2)

        # TODO: implement parameterizers

        training_data = None    # dummy
        sample = np.random.rand(sample_size, sample_size)   # dummy sample
        samples = np.array([sample])
        classes = ('authentic', 'forgery')

        # verifier_strategies = [CNNVerifierStrategy(), FixedThrVerifier()]
        verifier_strategies = [CNNVerifierStrategy()]
        run_verifier_strategies(verifier_strategies, training_data, samples, classes)


main()
