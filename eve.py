# Copyright 2023 Afshin Khadangi. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code has been adapted from 
# https://github.com/keras-team/keras/blob/v2.13.1/keras/optimizers/adam.py
# ==============================================================================
"""EVE optimiser implementation."""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Optimizer

class EVE(Optimizer):

    r"""Optimiser that implements the EVE algorithm.

    EVE optimisation is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments. The algorithm
    has dual learning rates.

    According to
    [Khadangi A., 2023](http://arxiv.org/),

    Args:
      learning_rate_1: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to `0.001`.
      learning_rate_2: A `tf.Tensor`, floating point value, a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use. The
        learning rate. Defaults to `0.0004`.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st short-term moment estimates. 
        Defaults to `0.9`.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st long-term moment estimates. 
        Defaults to `0.99`.
      beta_3: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the resultant 1st moment estimates. 
        Defaults to `0.5`.
      alpha: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 2nd moment estimates. Defaults to
        `0.999`.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        `1e-7`.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      {{base_optimizer_keyword_args}}

    Reference:
      - [Khadangi A., 2014](http://arxiv.org)
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). EVE utilises different decay values for sparse and dense gradients.
    """
    def __init__(
        self,
        learning_rate_1=0.001,
        learning_rate_2=0.0004,
        beta_1=0.9,
        beta_2=0.99,
        beta_3=0.5,
        alpha=0.999,
        epsilon=1e-7,
        amsgrad=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        jit_compile=True,
        name="EVE",
        **kwargs
    ):

        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )

        self._learning_rate = self._build_learning_rate(learning_rate_1)
        self.learning_rate_2 = self._build_learning_rate(learning_rate_2)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3
        self.alpha = alpha
        self.epsilon = epsilon
        self.amsgrad = amsgrad

    def build(self, var_list):
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        # short-term 1st momentum
        self._momentums_S = []
        # long-term 1st momentum
        self._momentums_L = []
        self._velocities_S = []
        self._velocities_L = []

        for var in var_list:
            self._momentums_S.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ms"
                )
            )
            self._momentums_L.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="ml"
                )
            )
            self._velocities_S.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="vs"
                )
            )
            self._velocities_L.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="vl"
                )
            )
        if self.amsgrad:
            self._velocity_hats = []
            for var in var_list:
                self._velocity_hats.append(
                    self.add_variable_from_reference(
                        model_variable=var, variable_name="vhat"
                    )
                )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        beta_1_power = None
        beta_2_power = None
        beta_3_power = None
        alpha_power = None
        lr = tf.cast(self._learning_rate, variable.dtype)
        lr_2 = tf.cast(self.learning_rate_2, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)
        beta_3_power = tf.pow(tf.cast(self.beta_3, variable.dtype), local_step)
        alpha_power = tf.pow(tf.cast(self.alpha, variable.dtype), local_step)


        var_key = self._var_key(variable)
        ms = self._momentums_S[self._index_dict[var_key]]
        ml = self._momentums_L[self._index_dict[var_key]]
        vs = self._velocities_S[self._index_dict[var_key]]
        vl = self._velocities_L[self._index_dict[var_key]]


        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            ms.assign_add(-ms * (1 - self.beta_1))
            ms.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_1), gradient.indices
                )
            )
            ml.assign_add(-ml * (1 - self.beta_2))
            ml.scatter_add(
                tf.IndexedSlices(
                    gradient.values * (1 - self.beta_2), gradient.indices
                )
            )
            vs.assign_add(-vs * (1 - self.alpha))
            vs.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values - K.sqrt(vl)) * (1 - self.alpha), gradient.indices,
                )
            )
            vl.assign_add(-vl * (1 - self.alpha))
            vl.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values - K.sqrt(vs)) * (1 - self.alpha),
                    gradient.indices,
                )
            )

            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, vs))
                vs = v_hat

            alpha_1 = lr * tf.sqrt(1 - alpha_power) / (1 - beta_1_power)
            alpha_2 = lr_2 * tf.sqrt(1 - beta_2_power) / (1 - beta_1_power)
            m_ = self.beta_3 * ms + (1 - self.beta_3) * ml
            eta = alpha_1 / (K.sqrt(vl) + self.epsilon)
            eta_2 = alpha_2 / (K.sqrt(vs) + self.epsilon)
            delta_theta = (- eta * m_ - eta_2 * m_) / 2
            variable.assign_add(delta_theta)
        else:
            # Dense gradients.
            ms.assign_add((gradient - ms) * (1 - self.beta_1))
            ml.assign_add((gradient - ml) * (1 - self.beta_2))
            vs.assign_add((tf.square(gradient - K.sqrt(vl)) - vs) * (1 - self.beta_2))
            vl.assign_add((tf.square(gradient - K.sqrt(vs)) - vl) * (1 - self.alpha))
            if self.amsgrad:
                v_hat = self._velocity_hats[self._index_dict[var_key]]
                v_hat.assign(tf.maximum(v_hat, vs))
                vs = v_hat
            alpha_1 = lr * tf.sqrt(1 - alpha_power) / (1 - beta_1_power)
            alpha_2 = lr_2 * tf.sqrt(1 - alpha_power) / (1 - beta_2_power)
            m_ = self.beta_3 * ms + (1 - self.beta_3) * ml
            eta = alpha_1 / (K.sqrt(vl) + self.epsilon)
            eta_2 = alpha_2 / (K.sqrt(vs) + self.epsilon)
            delta_theta = (- eta * m_ - eta_2 * m_) / 2
            variable.assign_add(delta_theta)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate_1": self._serialize_hyperparameter(
                    self._learning_rate
                ),
                "learning_rate_2": self._serialize_hyperparameter(
                    self.learning_rate_2
                ),
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_3": self.beta_3,
                "alpha": self.alpha,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        )
        return config

  
