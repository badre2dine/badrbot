"""


ref: https://www.tensorflow.org/agents/tutorials/2_environments_tutorial?hl=fr


"""
import abc
from re import I, S
import tensorflow as tf
import numpy
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.agents.ddpg import ddpg_agent
from enum import IntEnum


class Action(IntEnum):
    BUY = 0
    SELL = 1
    HOLD = 2


class FutureEnvirement(py_environment.PyEnvironment):
    def __init__(
        self,
        window_size: int,
        dataset: numpy.ndarray,
        target: int = 0,
        _initial_amount: float = 10000,
    ):
        self._window_size = window_size
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=numpy.int32, minimum=0, maximum=2, name="action"
        )
        self._observation_spec = array_spec.ArraySpec(
            shape=(window_size, dataset.shape[1]),
            dtype=numpy.float32,
            name="observation",
        )
        self._initial_amount = _initial_amount
        self._current_amount = None
        self._data_set = dataset
        self._target = target
        self._state = 0  # stand for the current day
        self._episode_ended = False
        self._current_operation = None
        self._postion_starting_amount = None
        self._total_profit = None
        self._fees = 0
        self._ratio_operation = 0.1
        self._tolerance = 0
        self._ratio_profit_per_postition = 0
        self._wanted_profit_per_postition = 0

    def action_spec(self) -> array_spec.BoundedArraySpec:

        return self._action_spec

    def observation_spec(self) -> array_spec.ArraySpec:

        return self._observation_spec

    def _get_state(self) -> numpy.ndarray:

        return numpy.stack(
            self._amount_history
            + self._data_set[self._state : self._state + self._window_size]
        )

    def _update_state(self, reward):

        self._total_reward += reward
        self._current_amount += reward
        self._state += 1
        self._amount_history.pop(0)
        self._amount_history.append(self._current_amount)
        # A negative amount of money or no more observation

        if (
            self._current_amount <= 0
            or self._state + self._window_size >= self._data_set.shape[0]
        ):
            self._episode_ended = True

    def _reset(self) -> ts.TimeStep:
        self._current_amount = self._initial_amount
        self._amount_history = [self._current_amount] * self._window_size
        self._state = 0
        self._episode_ended = False
        self._current_operation = Action.HOLD
        self._total_reward = 0

        return ts.restart(self._data_set[: self._window_size])

    def _step(self, action):

        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        reward = 0

        if action == Action.HOLD:
            if self._current_operation == Action.BUY:
                reward += (
                    self._data_set[self._state + self._window_size][self._target]
                    - self._data_set[self._state + self._window_size - 1][self._target]
                )
            elif self._current_operation == Action.SELL:
                reward += (
                    self._data_set[self._state + self._window_size - 1][self._target]
                    - self._data_set[self._state + self._window_size][self._target]
                )
            else:
                reward = 0

        elif action == Action.BUY:
            if self._current_operation == Action.HOLD:
                self._postion_starting_amount = self._current_amount
                self._current_operation = action
                reward += (
                    self._data_set[self._state + self._window_size][self._target]
                    - self._data_set[self._state + self._window_size - 1][self._target]
                )
                reward -= self._current_amount * self._ratio_operation * self._fees
            elif self._current_operation == Action.SELL:
                position_profit = (
                    self._current_amount - self._postion_starting_amount
                ) / self._postion_starting_amount
                self._current_operation = Action.HOLD
                reward = (
                    position_profit - self._wanted_profit_per_postition
                ) * self._postion_starting_amount
            else:
                reward = 0

        elif action == Action.SELL:
            if self._current_operation == Action.HOLD:
                self._postion_starting_amount = self._current_amount
                self._current_operation = action
                reward += (
                    self._data_set[self._state + self._window_size - 1][self._target]
                    - self._data_set[self._state + self._window_size][self._target]
                )
                reward -= self._current_amount * self._ratio_operation * self._fees
            elif self._current_operation == Action.BUY:
                position_profit = (
                    self._current_amount - self._postion_starting_amount
                ) / self._postion_starting_amount
                self._current_operation = Action.HOLD
                reward = (
                    position_profit - self._wanted_profit_per_postition
                ) * self._postion_starting_amount
            else:
                reward = 0
        else:
            raise ValueError("`action` should be 0, 1, 2 but : ", action)
        print(self._current_amount)
        self._update_state(reward)

        if self._episode_ended:
            # FIXME:  correct this case
            return ts.termination(
                self._data_set[self._state : self._state + self._window_size],
                reward=0,
            )
        else:
            return ts.transition(
                self._data_set[self._state : self._state + self._window_size],
                reward=reward,
                discount=1.0,
            )


def make_rnn_neural_network(fc_layer_params, lstm_size):

    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def lstm_layers(lstm_size):
        return [
            tf.keras.layers.LSTM(num_units, return_sequences=True)
            for num_units in lstm_size
        ]

    def dense_layers(fc_layer_params):
        return [
            tf.keras.layers.Dense(
                num_units,
                activation=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(
                    scale=2.0, mode="fan_in", distribution="truncated_normal"
                ),
            )
            for num_units in fc_layer_params
        ]

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.

    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2),
    )
    q_net = sequential.Sequential(dense_layers(fc_layer_params) + [q_values_layer])
    return


if __name__ == "__main__":
    data = numpy.random.uniform(1, 40, (5, 3)).astype(numpy.float32)

    environment = FutureEnvirement(3, data)

    utils.validate_py_environment(environment, episodes=5)
