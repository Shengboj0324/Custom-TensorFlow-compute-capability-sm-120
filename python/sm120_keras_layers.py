"""
High-Level Keras Layers for SM120 Optimized Operations
Provides seamless integration with TensorFlow/Keras ecosystem
with automatic fallback and gradient support.

Copyright 2024 - TensorFlow SM120 Optimization Project
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Union, List, Tuple, Any
import warnings

try:
    import sm120_ops

    SM120_AVAILABLE = True
except ImportError:
    SM120_AVAILABLE = False
    warnings.warn(
        "SM120 optimized operations not available. Using standard TensorFlow implementations."
    )


class SM120Layer(tf.keras.layers.Layer):
    """Base class for all SM120 optimized layers with automatic fallback."""

    def __init__(self, use_sm120: bool = True, fallback_on_error: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_sm120 = use_sm120 and SM120_AVAILABLE
        self.fallback_on_error = fallback_on_error
        self._sm120_enabled = self.use_sm120

    def _try_sm120_operation(self, sm120_func, fallback_func, *args, **kwargs):
        """Execute SM120 operation with automatic fallback."""
        if not self._sm120_enabled:
            return fallback_func(*args, **kwargs)

        try:
            return sm120_func(*args, **kwargs)
        except Exception as e:
            if self.fallback_on_error:
                warnings.warn(
                    f"SM120 operation failed, falling back to standard implementation: {e}"
                )
                self._sm120_enabled = False
                return fallback_func(*args, **kwargs)
            else:
                raise e


class SM120Dense(SM120Layer):
    """SM120 optimized dense (fully connected) layer with Tensor Core acceleration.

    This layer provides significant performance improvements for large matrix
    multiplications on RTX 50-series GPUs while maintaining full compatibility
    with standard tf.keras.layers.Dense.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the kernel weights.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output.
        kernel_constraint: Constraint function applied to the kernel weights.
        bias_constraint: Constraint function applied to the bias vector.
        use_sm120: Boolean, whether to use SM120 optimized kernels.
        fallback_on_error: Boolean, whether to fallback to standard ops on error.
    """

    def __init__(
        self,
        units: int,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
        bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
        use_sm120: bool = True,
        fallback_on_error: bool = True,
        **kwargs,
    ):
        super().__init__(use_sm120=use_sm120, fallback_on_error=fallback_on_error, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        if self.units <= 0:
            raise ValueError(f"Expected units > 0, got {units}")

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                f"A Dense layer can only be built with a floating-point dtype, got {dtype}"
            )

        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                f"The last dimension of the inputs to a Dense layer should be defined, got {input_shape}"
            )

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=2, axes={-1: last_dim})

        self.kernel = self.add_weight(
            "kernel",
            shape=[last_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[self.units],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None

        super().build(input_shape)

    def call(self, inputs):
        def sm120_matmul():
            return sm120_ops.advanced_matmul(inputs, self.kernel)

        def standard_matmul():
            return tf.linalg.matmul(inputs, self.kernel)

        outputs = self._try_sm120_operation(sm120_matmul, standard_matmul)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                f"The last dimension of the input shape must be defined, got {input_shape}"
            )
        return input_shape[:-1].concatenate(self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
                "use_sm120": self.use_sm120,
                "fallback_on_error": self.fallback_on_error,
            }
        )
        return config


class SM120Conv2D(SM120Layer):
    """SM120 optimized 2D convolution layer with advanced memory coalescing.

    Provides substantial performance improvements for convolution operations
    on RTX 50-series GPUs while maintaining compatibility with tf.keras.layers.Conv2D.

    Args:
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers.
        strides: An integer or tuple/list of 2 integers.
        padding: One of "valid" or "same" (case-insensitive).
        data_format: A string, one of "channels_last" (default) or "channels_first".
        dilation_rate: An integer or tuple/list of 2 integers.
        groups: A positive integer specifying the number of groups.
        activation: Activation function to use.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the kernel weights.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the kernel weights.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output.
        kernel_constraint: Constraint function applied to the kernel weights.
        bias_constraint: Constraint function applied to the bias vector.
        use_sm120: Boolean, whether to use SM120 optimized kernels.
        fallback_on_error: Boolean, whether to fallback to standard ops on error.
    """

    def __init__(
        self,
        filters: int,
        kernel_size: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: str = "valid",
        data_format: Optional[str] = None,
        dilation_rate: Union[int, Tuple[int, int]] = (1, 1),
        groups: int = 1,
        activation: Optional[Union[str, callable]] = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
        bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
        use_sm120: bool = True,
        fallback_on_error: bool = True,
        **kwargs,
    ):
        super().__init__(use_sm120=use_sm120, fallback_on_error=fallback_on_error, **kwargs)

        self.filters = filters
        self.kernel_size = tf.keras.utils.normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = tf.keras.utils.normalize_tuple(strides, 2, "strides")
        self.padding = tf.keras.utils.normalize_padding(padding)
        self.data_format = tf.keras.utils.normalize_data_format(data_format)
        self.dilation_rate = tf.keras.utils.normalize_tuple(dilation_rate, 2, "dilation_rate")
        self.groups = groups
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        if self.groups != 1:
            warnings.warn(
                "SM120Conv2D currently only supports groups=1. Using standard implementation."
            )
            self._sm120_enabled = False

        if self.dilation_rate != (1, 1):
            warnings.warn(
                "SM120Conv2D currently only supports dilation_rate=(1,1). Using standard implementation."
            )
            self._sm120_enabled = False

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                f"The number of input channels must be evenly divisible by the number of groups. "
                f"Received groups={self.groups}, but the input has {input_channel} channels."
            )
        if self.filters % self.groups != 0:
            raise ValueError(
                f"The number of filters must be evenly divisible by the number of groups. "
                f"Received groups={self.groups}, but filters={self.filters}."
            )

        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype,
            )
        else:
            self.bias = None

        channel_axis = self._get_channel_axis()
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_channel}
        )
        super().build(input_shape)

    def call(self, inputs):
        def sm120_conv2d():
            return sm120_ops.conv2d(
                inputs, self.kernel, strides=(1, *self.strides, 1), padding=self.padding.upper()
            )

        def standard_conv2d():
            return tf.nn.conv2d(
                inputs,
                self.kernel,
                strides=(1, *self.strides, 1),
                padding=self.padding.upper(),
                data_format="NHWC" if self.data_format == "channels_last" else "NCHW",
                dilations=[1, *self.dilation_rate, 1],
            )

        outputs = self._try_sm120_operation(sm120_conv2d, standard_conv2d)

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format="NHWC")

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    @property
    def rank(self):
        return 2

    def _get_channel_axis(self):
        if self.data_format == "channels_first":
            return 1
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs should be defined, "
                f"Found input shape: {input_shape}"
            )
        return int(input_shape.dims[channel_axis].value)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_last":
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = tf.keras.utils.conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0]] + new_space + [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = tf.keras.utils.conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i],
                )
                new_space.append(new_dim)
            return tf.TensorShape([input_shape[0], self.filters] + new_space)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "data_format": self.data_format,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
                "use_sm120": self.use_sm120,
                "fallback_on_error": self.fallback_on_error,
            }
        )
        return config


class SM120BatchNormalization(SM120Layer):
    """SM120 optimized batch normalization layer.

    Provides optimized batch normalization for RTX 50-series GPUs with improved
    memory access patterns and reduced kernel launch overhead.
    """

    def __init__(
        self,
        axis: int = -1,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        center: bool = True,
        scale: bool = True,
        beta_initializer: str = "zeros",
        gamma_initializer: str = "ones",
        moving_mean_initializer: str = "zeros",
        moving_variance_initializer: str = "ones",
        beta_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        gamma_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        beta_constraint: Optional[tf.keras.constraints.Constraint] = None,
        gamma_constraint: Optional[tf.keras.constraints.Constraint] = None,
        use_sm120: bool = True,
        fallback_on_error: bool = True,
        **kwargs,
    ):
        super().__init__(use_sm120=use_sm120, fallback_on_error=fallback_on_error, **kwargs)

        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.moving_mean_initializer = tf.keras.initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = tf.keras.initializers.get(moving_variance_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        if not input_shape.ndims:
            raise ValueError("Input has undefined rank.")
        ndims = len(input_shape)

        # Convert axis to positive
        if self.axis < 0:
            axis = ndims + self.axis
        else:
            axis = self.axis

        if axis < 0 or axis >= ndims:
            raise ValueError(f"Invalid axis: {self.axis}")

        param_shape = [input_shape[axis]]

        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
            )
        else:
            self.beta = None

        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=param_shape,
            initializer=self.moving_mean_initializer,
            trainable=False,
        )

        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=param_shape,
            initializer=self.moving_variance_initializer,
            trainable=False,
        )

        super().build(input_shape)

    def call(self, inputs, training=None):
        def sm120_batch_norm():
            if SM120_AVAILABLE and hasattr(sm120_ops, "batch_normalization"):
                return sm120_ops.batch_normalization(
                    inputs,
                    self.gamma,
                    self.beta,
                    self.moving_mean,
                    self.moving_variance,
                    self.epsilon,
                    training=training,
                )
            else:
                raise NotImplementedError("SM120 batch normalization not available")

        def standard_batch_norm():
            return tf.nn.batch_normalization(
                inputs,
                mean=self.moving_mean if not training else None,
                variance=self.moving_variance if not training else None,
                offset=self.beta,
                scale=self.gamma,
                variance_epsilon=self.epsilon,
            )

        return self._try_sm120_operation(sm120_batch_norm, standard_batch_norm)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "axis": self.axis,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                "gamma_initializer": tf.keras.initializers.serialize(self.gamma_initializer),
                "moving_mean_initializer": tf.keras.initializers.serialize(
                    self.moving_mean_initializer
                ),
                "moving_variance_initializer": tf.keras.initializers.serialize(
                    self.moving_variance_initializer
                ),
                "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
                "gamma_regularizer": tf.keras.regularizers.serialize(self.gamma_regularizer),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
                "use_sm120": self.use_sm120,
                "fallback_on_error": self.fallback_on_error,
            }
        )
        return config


class SM120MultiHeadAttention(SM120Layer):
    """SM120 optimized multi-head attention layer with Flash Attention.

    Provides memory-efficient attention computation optimized for RTX 50-series GPUs
    with substantial memory savings and performance improvements for transformer models.
    """

    def __init__(
        self,
        num_heads: int,
        key_dim: int,
        value_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        output_shape: Optional[Union[int, Tuple[int, ...]]] = None,
        attention_axes: Optional[Union[int, Tuple[int, ...]]] = None,
        kernel_initializer: str = "glorot_uniform",
        bias_initializer: str = "zeros",
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
        bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
        use_sm120: bool = True,
        fallback_on_error: bool = True,
        use_flash_attention: bool = True,
        **kwargs,
    ):
        super().__init__(use_sm120=use_sm120, fallback_on_error=fallback_on_error, **kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.dropout = dropout
        self.use_bias = use_bias
        self.output_shape = output_shape
        self.attention_axes = attention_axes
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.use_flash_attention = use_flash_attention

        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, list):
            if len(input_shape) != 3:
                raise ValueError("MultiHeadAttention expects exactly 3 inputs: [query, key, value]")
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape

        query_shape = tf.TensorShape(query_shape)
        key_shape = tf.TensorShape(key_shape)
        value_shape = tf.TensorShape(value_shape)

        query_dims = query_shape[-1]
        key_dims = key_shape[-1]
        value_dims = value_shape[-1]

        # Query projection
        self.query_dense = SM120Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_sm120=self.use_sm120,
            fallback_on_error=self.fallback_on_error,
            name="query",
        )

        # Key projection
        self.key_dense = SM120Dense(
            self.num_heads * self.key_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_sm120=self.use_sm120,
            fallback_on_error=self.fallback_on_error,
            name="key",
        )

        # Value projection
        self.value_dense = SM120Dense(
            self.num_heads * self.value_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_sm120=self.use_sm120,
            fallback_on_error=self.fallback_on_error,
            name="value",
        )

        # Output projection
        output_dim = self.output_shape if self.output_shape is not None else query_dims
        self.output_dense = SM120Dense(
            output_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
            use_sm120=self.use_sm120,
            fallback_on_error=self.fallback_on_error,
            name="output",
        )

        super().build(input_shape)

    def call(self, inputs, attention_mask=None, return_attention_scores=False, training=None):
        if isinstance(inputs, list):
            if len(inputs) != 3:
                raise ValueError("MultiHeadAttention expects exactly 3 inputs: [query, key, value]")
            query, key, value = inputs
        else:
            query = key = value = inputs

        # Project to multi-head representations
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Reshape for multi-head attention
        batch_size = tf.shape(query)[0]
        seq_len_q = tf.shape(query)[1]
        seq_len_k = tf.shape(key)[1]

        query = tf.reshape(query, [batch_size, seq_len_q, self.num_heads, self.key_dim])
        key = tf.reshape(key, [batch_size, seq_len_k, self.num_heads, self.key_dim])
        value = tf.reshape(value, [batch_size, seq_len_k, self.num_heads, self.value_dim])

        # Transpose for attention computation: [batch, heads, seq_len, dim]
        query = tf.transpose(query, [0, 2, 1, 3])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])

        def sm120_attention():
            if (
                self.use_flash_attention
                and SM120_AVAILABLE
                and hasattr(sm120_ops, "scaled_dot_product_attention")
            ):
                return sm120_ops.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    scale=1.0 / np.sqrt(float(self.key_dim)),
                    dropout_rate=self.dropout if training else 0.0,
                    attention_mask=attention_mask,
                )
            else:
                raise NotImplementedError("SM120 attention not available")

        def standard_attention():
            # Standard scaled dot-product attention
            scale = 1.0 / np.sqrt(float(self.key_dim))
            scores = tf.linalg.matmul(query, key, transpose_b=True) * scale

            if attention_mask is not None:
                scores += attention_mask * -1e9

            attention_weights = tf.nn.softmax(scores, axis=-1)

            if training and self.dropout > 0:
                attention_weights = tf.nn.dropout(attention_weights, rate=self.dropout)

            attention_output = tf.linalg.matmul(attention_weights, value)

            if return_attention_scores:
                return attention_output, attention_weights
            return attention_output

        if return_attention_scores:
            attention_output, attention_weights = self._try_sm120_operation(
                lambda: sm120_attention(), standard_attention
            )
        else:
            attention_output = self._try_sm120_operation(sm120_attention, standard_attention)

        # Transpose back and reshape
        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len_q, self.num_heads * self.value_dim]
        )

        # Final output projection
        output = self.output_dense(attention_output)

        if return_attention_scores:
            return output, attention_weights
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "value_dim": self.value_dim,
                "dropout": self.dropout,
                "use_bias": self.use_bias,
                "output_shape": self.output_shape,
                "attention_axes": self.attention_axes,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
                "bias_regularizer": tf.keras.regularizers.serialize(self.bias_regularizer),
                "activity_regularizer": tf.keras.regularizers.serialize(self.activity_regularizer),
                "kernel_constraint": tf.keras.constraints.serialize(self.kernel_constraint),
                "bias_constraint": tf.keras.constraints.serialize(self.bias_constraint),
                "use_sm120": self.use_sm120,
                "fallback_on_error": self.fallback_on_error,
                "use_flash_attention": self.use_flash_attention,
            }
        )
        return config


# Utility functions for creating SM120-optimized models
def create_sm120_transformer_encoder(
    vocab_size: int,
    max_length: int,
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int = 6,
    dropout_rate: float = 0.1,
    use_sm120: bool = True,
) -> tf.keras.Model:
    """Create a transformer encoder model using SM120 optimized layers.

    Args:
        vocab_size: Size of vocabulary
        max_length: Maximum sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ff_dim: Feed-forward network dimension
        num_layers: Number of transformer layers
        dropout_rate: Dropout rate
        use_sm120: Whether to use SM120 optimized layers

    Returns:
        Compiled Keras model
    """

    # Input layer
    inputs = tf.keras.Input(shape=(max_length,), dtype=tf.int32)

    # Embedding and positional encoding
    embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    positions = tf.keras.layers.Embedding(max_length, embed_dim)(
        tf.range(start=0, limit=max_length, delta=1)
    )
    x = embeddings + positions

    # Transformer layers
    for i in range(num_layers):
        # Multi-head attention
        attention_output = SM120MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            use_sm120=use_sm120,
            name=f"attention_{i}",
        )(x)
        attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

        # Feed-forward network
        ff_output = SM120Dense(ff_dim, activation="relu", use_sm120=use_sm120, name=f"ff1_{i}")(x)
        ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
        ff_output = SM120Dense(embed_dim, use_sm120=use_sm120, name=f"ff2_{i}")(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

    # Global average pooling and output
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = SM120Dense(vocab_size, activation="softmax", use_sm120=use_sm120, name="output")(x)

    model = tf.keras.Model(inputs, outputs, name="sm120_transformer_encoder")
    return model


# Register custom layers for serialization
tf.keras.utils.get_custom_objects().update(
    {
        "SM120Dense": SM120Dense,
        "SM120Conv2D": SM120Conv2D,
        "SM120BatchNormalization": SM120BatchNormalization,
        "SM120MultiHeadAttention": SM120MultiHeadAttention,
    }
)
