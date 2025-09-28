# custom_losses.py
import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss implementado como clase serializable.
    Se puede usar directamente al compilar el modelo y al cargarlo
    no requiere custom_objects, solo importar este archivo antes.
    """

    def __init__(
        self,
        gamma=2.0,
        alpha=0.75,
        reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name="binary_focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        return -tf.reduce_mean(w * tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config
