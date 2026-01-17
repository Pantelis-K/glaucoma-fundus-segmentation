from tensorflow.keras import layers, models


def double_conv_block(x, n_filters):
    """Two Conv2D + ReLU layers with He normal initialization."""
    x = layers.Conv2D(
        n_filters, 3, padding="same",
        activation="relu",
        kernel_initializer="he_normal")(x)
    x = layers.Conv2D(
        n_filters, 3, padding="same",
        activation="relu",
        kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters, dropout_rate=0.3):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(dropout_rate)(p)
    return f, p


def upsample_block(x, conv_features, n_filters, dropout_rate=0.3):
    x = layers.Conv2DTranspose(
        n_filters, 3, strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv_features])
    x = layers.Dropout(dropout_rate)(x)
    x = double_conv_block(x, n_filters)
    return x


def build_UNET(
    input_shape,
    n_classes=1,
    name="UNET"
):
    """
    Builds U-Net variant for fundus image segmentation.

    Parameters
    ----------
    input_shape : tuple
        (H, W, C)
    n_classes : int
        Number of output channels (1 for binary segmentation)
    name : str
        Model name
    """

    inputs = layers.Input(shape=input_shape)

    # -------- encoder --------
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # -------- bottleneck -----
    bottleneck = double_conv_block(p4, 1024)

    # -------- decoder --------
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(
        n_classes, 1, padding="same",
        activation="sigmoid")(u9)

    return models.Model(inputs, outputs, name=name)
