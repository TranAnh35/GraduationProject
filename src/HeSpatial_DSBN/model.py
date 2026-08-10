"""
HeSpatial-DSBN Model Architecture:
Heterogeneous Spatial Context Autoencoder with Multi-Domain Specific Batch Normalization,
Sensor-Specific Adapters, GRU Bottleneck, K-Sparse Activation, and GRL Domain Classifier.

Standalone module - no external dependencies.
"""

from __future__ import annotations
from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import layers

from . import config
from .losses import GradientReversalLayer


# =============================================================================
# MULTI-DOMAIN BATCH NORMALIZATION (MultiDomainBatchNorm)
# =============================================================================

class MultiDomainBatchNorm(layers.Layer):
    """
    Multi-Domain Specific Batch Normalization (DSBN) supporting N domains.
    Maintains N dedicated BatchNormalization instances with private mean/variance and affine parameters.
    """
    def __init__(self, num_domains: int = config.NUM_DOMAINS, name: str = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_domains = num_domains
        self.bns = [
            layers.BatchNormalization(name=f"{name}_d{d}" if name else f"bn_d{d}")
            for d in range(num_domains)
        ]

    def call(self, inputs, training=None) -> tf.Tensor:
        x, domain_id = inputs
        d_val = tf.reduce_min(tf.cast(domain_id, tf.int32))
        
        # Route dynamically to the BN instance corresponding to d_val
        def route_bn(idx: int):
            if idx >= self.num_domains - 1:
                return self.bns[self.num_domains - 1](x, training=training)
            return tf.cond(
                tf.equal(d_val, idx),
                lambda: self.bns[idx](x, training=training),
                lambda: route_bn(idx + 1)
            )

        return route_bn(0)

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({"num_domains": self.num_domains})
        return cfg


def apply_multidomain_bn(x: tf.Tensor, domain_id: tf.Tensor, name: str) -> tf.Tensor:
    """Utility function to apply Multi-Domain DSBN or standard BN."""
    if config.DSBN_ENABLED:
        return MultiDomainBatchNorm(num_domains=config.NUM_DOMAINS, name=name)([x, domain_id])
    else:
        return layers.BatchNormalization(name=name)(x)


# =============================================================================
# SENSOR-SPECIFIC ADAPTOR (Heterogeneous Sensor-Specific Projection)
# =============================================================================

class SensorAdapter(layers.Layer):
    """
    Sensor-Specific 1D Convolution Adapter.
    Projects heterogeneous sensor inputs (Hall, Coil, Diffensor) to unified feature depth.
    """
    def __init__(self, proj_channels: int = config.ADAPTER_PROJ_CHANNELS, name: str = "sensor_adapter", **kwargs):
        super().__init__(name=name, **kwargs)
        self.proj_channels = proj_channels
        self.conv_proj = layers.Conv1D(
            filters=proj_channels,
            kernel_size=1,
            padding="same",
            activation="swish",
            name=f"{name}_conv1d"
        )
        self.ln = layers.LayerNormalization(name=f"{name}_ln")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv_proj(inputs)
        return self.ln(x)


# =============================================================================
# BUILDING BLOCKS (ResBlock & K-Sparse Activation)
# =============================================================================

def _res_block_sepconv1d(
    x: tf.Tensor,
    domain_id: tf.Tensor,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    name: str,
    stride: int = 1,
) -> tf.Tensor:
    """Residual Block with Separable Conv1D, MultiDomainBatchNorm, and Swish Activation."""
    y = layers.SeparableConv1D(
        filters, kernel_size, strides=stride, padding="same",
        dilation_rate=dilation_rate, name=f"{name}_sepconv1",
    )(x)
    y = apply_multidomain_bn(y, domain_id, name=f"{name}_dsbn1")
    y = layers.Activation("swish", name=f"{name}_act1")(y)
    
    if dropout_rate and dropout_rate > 0:
        y = layers.Dropout(dropout_rate, name=f"{name}_drop1")(y)

    y = layers.SeparableConv1D(
        filters, kernel_size, padding="same", name=f"{name}_sepconv2",
    )(y)
    y = apply_multidomain_bn(y, domain_id, name=f"{name}_dsbn2")
    y = layers.Activation("swish", name=f"{name}_act2")(y)

    if x.shape[-1] != filters or stride > 1:
        skip = layers.Conv1D(filters, 1, strides=stride, padding="same", name=f"{name}_skip")(x)
        skip = apply_multidomain_bn(skip, domain_id, name=f"{name}_skip_dsbn")
        skip = layers.Activation("swish", name=f"{name}_skip_act")(skip)
    else:
        skip = x

    return layers.Add(name=f"{name}_add")([y, skip])


def k_sparse_layer(tensor: tf.Tensor, k: int = 16) -> tf.Tensor:
    """Zeroes out activations outside top-k magnitudes."""
    values, _ = tf.math.top_k(tf.abs(tensor), k=k)
    min_values = tf.reduce_min(values, axis=-1, keepdims=True)
    mask = tf.cast(tf.abs(tensor) >= min_values, tensor.dtype)
    return tensor * mask


# =============================================================================
# ENCODER & DECODER MODELS
# =============================================================================

def build_hespatial_encoder(
    input_shape: tuple,
    filters_start: int = config.FILTERS_START,
    latent_dim: int = config.LATENT_DIM,
    dropout_rate: float = config.DROPOUT_RATE,
    name: str = "hespatial_encoder",
) -> tf.keras.Model:
    """Builds Shared Spatial Context Encoder."""
    inp = layers.Input(shape=input_shape, name="encoder_input")
    domain_id = layers.Input(shape=(), dtype="int32", name="domain_id")
    
    x = layers.LayerNormalization(name="ln_in")(inp)
    x = layers.Conv1D(filters_start, 1, padding="same", name="proj", activation="swish")(x)

    x = _res_block_sepconv1d(x, domain_id, filters_start, 7, 1, dropout_rate, "enc_b0", stride=2)
    x = _res_block_sepconv1d(x, domain_id, filters_start * 2, 7, 1, dropout_rate, "enc_b1", stride=2)
    x = _res_block_sepconv1d(x, domain_id, filters_start * 4, 5, 2, dropout_rate, "enc_b2", stride=1)

    x = layers.GRU(latent_dim, return_sequences=True, name="gru_bottleneck")(x)
    k_val = max(1, int(latent_dim * config.K_SPARSE_RATIO))
    x = layers.Lambda(lambda t: k_sparse_layer(t, k=k_val), name="k_sparse_activation")(x)
    x = layers.LayerNormalization(name="ln_latent")(x)
    
    return tf.keras.Model([inp, domain_id], x, name=name)


def build_hespatial_decoder(
    input_shape: tuple,
    filters_start: int = config.FILTERS_START,
    output_channels: int = 1,
    name: str = "hespatial_decoder",
) -> tf.keras.Model:
    """Builds Spatial Context Transposed Conv Decoder."""
    dec_in = layers.Input(shape=input_shape, name="decoder_input")
    domain_id = layers.Input(shape=(), dtype="int32", name="domain_id")
    
    x = layers.Conv1DTranspose(filters_start * 2, 3, strides=2, padding="same", name="up1")(dec_in)
    x = apply_multidomain_bn(x, domain_id, name="up1_dsbn")
    x = layers.Activation("swish")(x)
    
    x = layers.Conv1DTranspose(filters_start, 3, strides=2, padding="same", name="up2")(x)
    x = apply_multidomain_bn(x, domain_id, name="up2_dsbn")
    x = layers.Activation("swish")(x)

    out = layers.TimeDistributed(layers.Dense(output_channels, dtype="float32"), name="time_out")(x)
    return tf.keras.Model([dec_in, domain_id], out, name=name)


def build_domain_classifier(
    input_shape: tuple,
    num_domains: int = config.NUM_DOMAINS,
    hidden: int = config.DOMAIN_HIDDEN,
    name: str = "domain_classifier",
) -> tf.keras.Model:
    """Multi-Class / Binary Domain Discriminator with GAP and Dense layers."""
    inp = layers.Input(shape=input_shape, name="domain_input")
    x = layers.GlobalAveragePooling1D(name="gap")(inp)
    x = layers.Dense(hidden, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-3), name="domain_fc1")(x)
    x = layers.Dropout(0.3, name="domain_drop")(x)
    
    if num_domains == 2:
        out = layers.Dense(1, activation="sigmoid", name="domain_output")(x)
    else:
        out = layers.Dense(num_domains, activation="softmax", name="domain_output")(x)
        
    return tf.keras.Model(inp, out, name=name)


# =============================================================================
# FULL HESPATIAL-DSBN MODEL ASSEMBLY
# =============================================================================

def build_hespatial_dsbn_model(
    input_shape: tuple = (config.TIME_SAMPLES, config.NUM_CONTEXT_CHANNELS),
    filters_start: int = config.FILTERS_START,
    latent_dim: int = config.LATENT_DIM,
    dropout_rate: float = config.DROPOUT_RATE,
    grl_lambda: float = 1.0,
    domain_hidden: int = config.DOMAIN_HIDDEN,
    num_domains: int = config.NUM_DOMAINS,
    name: str = "hespatial_dsbn",
) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """
    Assembles full HeSpatial-DSBN architecture:
    Returns (full_model, encoder, decoder, domain_head).
    """
    encoder = build_hespatial_encoder(input_shape, filters_start, latent_dim, dropout_rate)
    enc_out_shape = encoder.output_shape[1:]
    
    decoder = build_hespatial_decoder(enc_out_shape, filters_start)
    domain_head = build_domain_classifier(enc_out_shape, num_domains=num_domains, hidden=domain_hidden)
    grl = GradientReversalLayer(lambda_value=grl_lambda, name="grl")
    
    x_s = layers.Input(shape=input_shape, name="source_input")
    x_t = layers.Input(shape=input_shape, name="target_input")
    d_id_s_in = layers.Input(shape=(), dtype="int32", name="domain_id_source")
    d_id_t_in = layers.Input(shape=(), dtype="int32", name="domain_id_target")

    z_s = encoder([x_s, d_id_s_in])
    z_t = encoder([x_t, d_id_t_in])
    
    y_s = decoder([z_s, d_id_s_in])
    y_t = decoder([z_t, d_id_t_in])
    
    z_s_grl = grl(z_s)
    z_t_grl = grl(z_t)
    z_concat = layers.Concatenate(axis=0, name="concat_latent")([z_s_grl, z_t_grl])
    d_logits = domain_head(z_concat)
    
    full_model = tf.keras.Model(
        inputs=[x_s, x_t, d_id_s_in, d_id_t_in],
        outputs=[y_s, y_t, d_logits, z_s, z_t],
        name=name
    )
    
    return full_model, encoder, decoder, domain_head


def build_inference_model(
    encoder: tf.keras.Model,
    decoder: tf.keras.Model,
    domain_id: int = 0,
    name: str = "hespatial_inference",
) -> tf.keras.Model:
    """Builds single-input inference model with fixed domain_id."""
    inp = layers.Input(shape=encoder.input_shape[0][1:], name="inference_input")
    d_id = layers.Lambda(
        lambda x: tf.fill(tf.shape(x)[:1], tf.constant(domain_id, dtype=tf.int32)),
        name="d_id_fixed"
    )(inp)
    
    z = encoder([inp, d_id])
    out = decoder([z, d_id])
    return tf.keras.Model(inp, out, name=name)
