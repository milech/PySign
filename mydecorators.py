# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
# - decorators used in other files
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

import time
from matplotlib import pyplot as plt


def timer4main(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        total = time.time() - start
        print(f"Execution time of the whole programme is: {total} s\n")
        return rv
    return wrapper


def timer(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        rv = func(self, *args, **kwargs)
        total = time.time() - start
        print(f"Execution time for {type(self)} is: {total} s\n")
        return rv
    return wrapper


def printall2console(func):
    def wrapper(*args, **kwargs):
        rv = func(*args, **kwargs)
        try:
            for no, result in enumerate(rv):
                print(f"Parameter {no} is {result}\n")
        except BaseException as err:
            print(f"This particular result cannot be printed to the console. Reason: {err}.\n")
        return rv
    return wrapper


def printcost2console(func):
    def wrapper(*args, **kwargs):
        rv = func(*args, **kwargs)
        try:
            print(f"Parameter {len(rv)-1} is {rv[-1]}\n")
        except BaseException as err:
            print(f"This particular result cannot be printed to the console. Reason: {err}.\n")
        return rv
    return wrapper


def plotresults(func):
    def wrapper(*args, **kwargs):
        rv = func(*args, **kwargs)
        try:
            _, (ax1, ax2) = plt.subplots(2, 1, sharex='col', num=rv[0], figsize=(5, 15))
            ax1.imshow(rv[1], interpolation='nearest')
            ax1.set_title('Cost matrix')
            ax2.imshow(rv[2], interpolation='nearest')
            ax2.set_title('Distance matrix')
            plt.suptitle(f'Total cost: {rv[3]:.2f}')
            plt.show()
        except BaseException as err:
            print(f"This particular result cannot be plotted. Reason: {err}.\n")
        return rv
    return wrapper
