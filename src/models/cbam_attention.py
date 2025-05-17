import tensorflow as tf
from keras import layers

from src.models.attention_module import AttentionModule

class CBAMAttention(AttentionModule):
    """Convolutional Block Attention Module (CBAM)."""
    
    def __init__(self, attention_ratio: int = 8):
        self.attention_ratio = attention_ratio
        
    def apply(self, input_feature: tf.Tensor) -> tf.Tensor:
        """Apply CBAM attention to input feature."""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(input_feature)
        max_pool = layers.GlobalMaxPooling2D()(input_feature)
        
        avg_pool = layers.Reshape((1, 1, input_feature.shape[-1]))(avg_pool)
        max_pool = layers.Reshape((1, 1, input_feature.shape[-1]))(max_pool)
        
        shared_dense_1 = layers.Dense(
            input_feature.shape[-1] // self.attention_ratio, 
            activation='relu',
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        
        shared_dense_2 = layers.Dense(
            input_feature.shape[-1],
            kernel_initializer='he_normal',
            use_bias=True,
            bias_initializer='zeros'
        )
        
        avg_pool = shared_dense_1(avg_pool)
        avg_pool = shared_dense_2(avg_pool)
        
        max_pool = shared_dense_1(max_pool)
        max_pool = shared_dense_2(max_pool)
        
        cbam_feature = layers.Add()([avg_pool, max_pool])
        cbam_feature = layers.Activation('sigmoid')(cbam_feature)
        
        # Apply channel attention
        channel_attention = layers.Multiply()([input_feature, cbam_feature])
        
        # Spatial attention
        avg_pool_spatial = layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True)
        )(channel_attention)
        
        max_pool_spatial = layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True)
        )(channel_attention)
        
        spatial_concat = layers.Concatenate()([avg_pool_spatial, max_pool_spatial])
        
        spatial_attention = layers.Conv2D(
            1, kernel_size=(7, 7), padding='same', 
            activation='sigmoid',
            kernel_initializer='he_normal'
        )(spatial_concat)
        
        # Apply spatial attention
        return layers.Multiply()([channel_attention, spatial_attention]) 