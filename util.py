"""
MIT License

Copyright (c) 2022 JAKE LEVI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import pickle
import traceback
import datetime
import time
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "Results")

class Result:
    def __init__(self, filename, data=None):
        self._data = data
        self._filename = filename

    def get_data(self):
        return self._data

    def get_context(self, save=True, suppress_exceptions=False):
        return ResultSavingContext(self, save, suppress_exceptions)

    def save(self):
        print("\nSaving results data to \"%s\"..." % self._filename)
        if not os.path.isdir(os.path.dirname(self._filename)):
            os.makedirs(os.path.dirname(self._filename))
        with open(self._filename, "wb") as f:
            pickle.dump(self._data, f)

    def load(self):
        print("Loading results data from \"%s\"..." % self._filename)
        with open(self._filename, "rb") as f:
            self._data = pickle.load(f)
        return self._data

class ResultSavingContext:
    def __init__(self, result, save, suppress_exceptions):
        self._result = result
        self._save = save
        self._suppress_exceptions = suppress_exceptions

    def __enter__(self):
        return self._result

    def __exit__(self, *args):
        if self._save:
            self._result.save()
        if self._suppress_exceptions:
            return True

class ExceptionContext:
    def __init__(self, suppress_exceptions=True, printer=None):
        self._suppress_exceptions = suppress_exceptions
        if printer is None:
            printer = Printer()
        self._print = printer

    def __enter__(self):
        return

    def __exit__(self, *args):
        if args[0] is not None:
            self._print("%s: An exception occured:" % datetime.datetime.now())
            self._print("".join(traceback.format_exception(*args)))
            if self._suppress_exceptions:
                self._print("Suppressing exception and continuing...")
                return True

class Printer:
    def __init__(
        self,
        output_filename=None,
        output_dir=None,
        print_to_console=True,
    ):
        if output_filename is not None:
            if output_dir is None:
                output_dir = RESULTS_DIR

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(output_dir, output_filename)
            self._file = open(output_path, "w")
        else:
            self._file = None

        self._print_to_console = print_to_console

    def __call__(self, *args, **kwargs):
        self.print(*args, **kwargs)

    def print(self, *args, **kwargs):
        if self._print_to_console:
            print(*args, **kwargs)
        if self._file is not None:
            print(*args, **kwargs, file=self._file)

    def close(self):
        if self._file is not None:
            self._file.close()

class Seeder:
    def __init__(self):
        self._used_seeds = set()

    def get_seed(self, *args):
        seed = sum((i + 1) * ord(c) for i, c in enumerate(str(args)))
        while seed in self._used_seeds:
            seed += 1

        self._used_seeds.add(seed)
        return seed

    def get_rng(self, *args):
        seed = self.get_seed(*args)
        rng = np.random.default_rng(seed)
        return rng

class Timer:
    def __init__(self, name=None, printer=None):
        self._name = name
        if printer is None:
            printer = Printer()
        self._print = printer
        self._t0 = time.perf_counter()

    def time_taken(self):
        t1 = time.perf_counter()
        return t1 - self._t0

    def print_time_taken(self):
        t = self.time_taken()
        if self._name is None:
            prefix = "Time taken"
        else:
            prefix = "Time taken for %s" % self._name

        if t > 60:
            m, s = divmod(t, 60)
            self._print("%s = %i minutes %.1f seconds" % (prefix, m, s))
        else:
            self._print("%s = %.3f s" % (prefix, t))

    def __enter__(self):
        return

    def __exit__(self, *args):
        self.print_time_taken()

def clean_filename(filename_str, allowed_non_alnum_chars="-_.,"):
    filename_str_clean = "".join(
        c if (c.isalnum() or c in allowed_non_alnum_chars) else "_"
        for c in str(filename_str)
    )
    return filename_str_clean

def is_numeric(x):
    return any(isinstance(x, t) for t in [int, float, np.number])

def numpy_set_print_options():
    np.set_printoptions(
        precision=3,
        linewidth=10000,
        suppress=True,
        threshold=10000,
    )
