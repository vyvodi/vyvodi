from vyvodi.layers.dense_hierarchical import DenseHierarchical

import tensorflow as tf

def test_DenseHierarchical():
    """Test the DenseHierarchical layer."""

    # Create a random effects layer.
    layer = DenseHierarchical(
        n_units=10,
        n_samples=2,
        activation=tf.nn.relu,
        use_bias=True,
        activity_regularizer=None,
    )

    # Create a random input.
    x = tf.keras.layers.Input(shape=(10,), dtype=tf.float32)
    category = tf.keras.layers.Input(shape=(), dtype=tf.int32)

    output = layer([x, category])

    assert output.shape == (None, 10)
    assert output.dtype == tf.float32
