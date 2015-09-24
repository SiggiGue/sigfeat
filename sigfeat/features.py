"""
Feature classes
===============


"""
import numpy as np
from scipy.fftlib import rfft as _rfft


class Feature(object):
    output_label_list = None
    input_feature_label_list = None

    @property
    def labels(self):
        return self.output_label_list

    def process(self, **kwargs):
        return list()


class Signal(Feature):
    output_label_list = ['signal', ]
    input_label_list = None


class AbsSignal(Feature):
    output_label_list = ['abs_signal', ]
    input_label_list = ['signal', ]

    def process(self, signal):
        return np.abs(signal)


class SquaredSignal(Feature):
    output_label_list = ['squared_signal', ]
    input_label_list = ['signal', ]

    def process(self, signal):
        return signal * signal


class RFFT(Feature):
    output_label_list = ['rfft', ]
    input_label_list = ['signal', ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def process(self, signal):
        return _rfft(signal, *self.args, **self.kwargs)


class Centroid(Feature):
    output_label_list = ['centroid', ]
    input_label_list = ['rfft', ]

    def process(self, rfft):
        return np.sum(np.arange(1, len(rfft) * rfft)) / np.sum(rfft)


class CrestFactor(Feature):
    output_label_list = ['crest_factor', ]
    input_label_list = ['signal', ]

    def process(self, signal):
        return np.max(np.abs(signal)) / np.sqrt(np.mean(signal*signal))
