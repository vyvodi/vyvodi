from .dense_hierarchical import DenseHierarchical

import tensorflow as tf
import tensorflow_probability as tfp

from typing import List, Tuple, Union

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfb = tfp.bijectors

class MLPHierarchical(tfkl.Layer):
    """MLP layer with random `kernel` and `bias` which are sampled from
    a shared normal distribution.

    This layers uses variational inference to approximate the posterior
    distribution of the random effects.

    """
    def __init__(
        self, units: List[int], 
        n_samples: int, 
        kl_weight: float = None,
        kl_use_exact: bool = False,
        activation = 'relu',
        final_activation = None,
        use_bias: bool = True,
        activity_regularizer=None,
        **kwargs
    ):
        """Create a random effect layer with the specified number of units 
        and categories.

        Args:
            units: Sequential list of units in the random effect layer.
            n_categories: Number of categories in the random effect layer.
            kl_weight: Weight of the KL divergence term in the loss function.
            kl_use_exact: Whether to use the exact KL divergence or 
                approximate KL divergence.
            activation: Activation function to use.
            final_activation: Activation function to use for the final layer.
            use_bias: Whether to use a bias term.
            activity_regularizer: Regularizer function for the output.
            **kwargs: Extra arguments forwards to `tf.keras.layers.Layer`.

        """
        super().__init__(
            activity_regularizer=tfk.regularizers.get(activity_regularizer),
            **kwargs
        )
        
        self._sublayers = []

        for i, unit in enumerate(units[:-1]):
            self._sublayers.append(
                DenseHierarchical(
                    unit, n_samples, 
                    kl_weight=kl_weight,
                    kl_use_exact=kl_use_exact,
                    activation=activation,
                    use_bias=use_bias,
                    name=f'dense_{i}'
                )
            )
        
        self._sublayers.append(
            DenseHierarchical(
                units[-1], n_samples,
                kl_weight=kl_weight,
                kl_use_exact=kl_use_exact,
                activation=final_activation,
                use_bias=use_bias,
                name=f'dense_{len(units)-1}'
            )
        )
    
    def call(self, inputs, **kwargs):
        for layer in self._sublayers:
            inputs = layer(inputs)
        
        return inputs
