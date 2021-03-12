import numpy as np
from typing import Callable


# Edit out later
def learning_rule_(_):
    return 0.05


class HyperParameters:
    """
    Parameters that can be configured to adjust model performance.
    """
    data_type = np.float64
    max_range = np.float64(.5)
    learning_rule = learning_rule_


del learning_rule_


def bind(inst, f: Callable[[object], HyperParameters.data_type], name=None) -> \
        Callable[[object], HyperParameters.data_type]:
    if name is None:
        name = f.__name__
    bounded_method = f.__get__(inst, inst.__class__)
    setattr(inst, name, bounded_method)
    return bounded_method


class Neuron:
    """
    The most atomic piece of the neural network.
    """

    def __init__(self, min_: HyperParameters.data_type, max_: HyperParameters.data_type,
                 max_range: HyperParameters.data_type = HyperParameters.max_range,
                 learning_rule: Callable[[object], HyperParameters.data_type] = HyperParameters.learning_rule) -> None:
        """
        Constructs a neuron.

        PARAMETERS
        --------
        min_
            Minimum activation value
        max_
            maximum activation value
        max_range
            Maximum difference between min_ and max_
        learning_rule
            Method called to determine the learning rate
        RETURNS
        --------
        None
        """
        if max_range is None:
            max_range = max_-min_
        self.min_, self.max_ = min_, max_
        self.max_range = max_range
        self.learning_rule = None
        bind(self, learning_rule, name='learning_rule')

    @property
    def min_(self):
        return self._min_

    @min_.setter
    def min_(self, value):
        self._min_ = 0 if value < 0 else value
        if self.max_-value > self.max_range:
            val = self._min_+self.max_range
            self.max_ = 1 if val > 1 else val

    @property
    def max_(self):
        return self._max_

    @max_.setter
    def max_(self, value):
        self._max_ = 1 if value > 1 else value
        if value-self.min_ > self.max_range:
            val = value-self.max_range
            self.min_ = 0 if val < 0 else val

    @property
    def max_range(self):
        return self._max_range

    @max_range.setter
    def max_range(self, value):
        if not 0 < value < 1:
            raise ValueError('max_range must be within range (0,1).')
        self._max_range = value

    def activate(self, value: HyperParameters.data_type) -> bool:
        """
        Checks if the given value activates the function.

        PARAMETERS
        --------
        value
            Value to check to see if it activates the neuron.
        RETURNS
        --------
        bool
            True if the neuron activates, False otherwise
        """
        return self.min_ <= value <= self.max_

    def adjust(self, value: HyperParameters.data_type) -> None:
        """
        Adjusts parameters to be more within the area of given a value.

        PARAMETERS
        --------
        value
            Value to adjust to.
        RETURNS
        --------
        None
        """
        # Make sure given value is in range (0,1)
        value = 0 if value < 0 else (1 if value > 1 else value)
        if value < self.min_:
            self.min_ -= self.learning_rule()
        if value > self.max_:
            self.max_ += self.learning_rule()


class Layer:
    def __init__(self, neuron_count, intake=None, output=None):
        self.neuron_count = neuron_count
        self.intake = intake
        self.output = output

    @property
    def neuron_count(self):
        return self._neuron_count

    @neuron_count.setter
    def neuron_count(self, value):
        self._neuron_count = value


class Connection:
    def __init__(self, intake=None, output=None):
        self.intake = [] if intake is None else intake
        self.output = [] if output is None else output

    @property
    def intake(self):
        return self._intake

    @intake.setter
    def intake(self, value):
        self._intake = value
        value.output = self

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value
        value.intake = self

    def add_intake(self, neuron):
        self.intake.append(neuron)

    def add_intakes(self, neurons):
        self.intake.extend(neurons)

    def add_output(self, neuron):
        self.output.append(neuron)

    def add_outputs(self, neurons):
        self.output.extend(neurons)
