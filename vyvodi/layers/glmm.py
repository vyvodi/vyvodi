from typing import List
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.layers.dense_variational_v2 import (
    _make_kl_divergence_penalty
)

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors


class RandomEffects(tfkl.Layer):
    """Dense layer with random `kernel` and `bias` which are sampled from
    a shared normal distribution.

    This layers uses variational inference to approximate the posterior
    distribution of the random effects.

    """
    def __init__(
        self, n_units, n_samples, 
        kl_weight=None,
        kl_use_exact=False,
        activation=None,
        use_bias=True,
        activity_regularizer=None,
        **kwargs
    ):
        """Create a random effect layer with the specified number of units 
        and categories.

        Args:
            n_units: Number of units in the random effect layer.
            n_categories: Number of categories in the random effect layer.
            kl_weight: Weight of the KL divergence term in the loss function.
            kl_use_exact: Whether to use the exact KL divergence or approximate
                KL divergence.
            activation: Activation function to use.
            use_bias: Whether to use a bias term.
            activity_regularizer: Regularizer function for the output.
            **kwargs: Extra arguments forwards to `tf.keras.layers.Layer`.

        """
        super().__init__(
            activity_regularizer=tfk.regularizers.get(activity_regularizer),
            **kwargs
        )
        self.n_units = int(n_units)
        self.n_samples = int(n_samples)
        self.activation = tfk.activations.get(activation)
        self.use_bias = use_bias
        
        self._kl_divergence_fn = _make_kl_divergence_penalty(
            kl_use_exact, weight=kl_weight
        )

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])

        if last_dim is None:
            raise ValueError(
                'The last dimension of the inputs to `RandomEffects` '
                'must be defined. Found `None`.'
            )

        n_priors = (last_dim + self.use_bias) * self.n_units
        self._prior = tfk.Sequential([
            tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n_priors)),
            tfpl.IndependentNormal((self.n_units, last_dim + self.use_bias))
        ])

        n_posteriors = n_priors * (self.n_samples + 1)
        self._posterior = tfk.Sequential([
            tfpl.VariableLayer(
                tfpl.IndependentNormal.params_size(n_posteriors)
            ),
            tfpl.IndependentNormal(
                (self.n_samples + 1, self.n_units, last_dim + self.use_bias)
            )
        ]) # mean-field approximation

        self.built = True

    def call(self, inputs, **kwargs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        x, category = self._parse_inputs(inputs)

        q = self._posterior(x)
        p = self._prior(x)
        self.add_loss(self._kl_divergence_fn(q, p))

        w = tf.convert_to_tensor(q)
        w = tf.gather(w, category, axis=0)
        
        if self.use_bias:
            w, b = tf.split(w, (self.n_units, 1), axis=2)
        else:
            b = tf.zeros((self.n_units, 1), dtype=dtype)

        outputs = tf.squeeze(
            tf.matmul(w, tf.expand_dims(x, axis=1), transpose_b=True) + b
        )

        if self.activation is not None:
            outputs = self.activation(outputs)
        
        return outputs

    def _parse_inputs(self, inputs):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

        if isinstance(inputs, (list, tuple)):
            x, category = inputs
            if not isinstance(category, tf.Tensor):
                category = tf.convert_to_tensor(category)
        
        # TODO: add support for other types of inputs
        else:
            raise ValueError(
                '`RandomEffects` expects a list or tuple of two tensors: '
                '`x` and `category`.'
            )

        x = tf.cast(x, dtype)
        category = tf.cast(category, tf.int32)
        
        return x, category