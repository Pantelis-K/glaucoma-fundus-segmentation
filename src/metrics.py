import tensorflow as tf

def dice_coef(y_true, y_pred): 
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-6) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-6) 

def dice_loss(y_true, y_pred): 
    return -dice_coef(y_true, y_pred)

def log_dice_loss(y_true, y_pred):
    return -tf.math.log(dice_coef(y_true, y_pred))

def iou(y_true, y_pred, threshold=0.5):                                                             
    y_true = tf.reshape(y_true, [-1])                                                               
    y_true = tf.cast(y_true, tf.float32)                                                            
    y_pred = tf.cast(y_pred > threshold, tf.float32)                                                 
    y_pred = tf.reshape(y_pred, [-1])                                                               
    intersection = tf.reduce_sum(y_true * y_pred)                                                     
    union = tf.reduce_sum(tf.cast(y_true + y_pred > 0, tf.float32))                                 
    return intersection / union


def sobel_edges(x):
    # Ensure float32 and shape [B, H, W, 1]
    x = tf.cast(x, tf.float32)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, axis=-1)

    Gx = tf.constant([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]], dtype=tf.float32, shape=(3, 3, 1, 1))
    
    Gy = tf.constant([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]], dtype=tf.float32, shape=(3, 3, 1, 1))
    
    grad_x = tf.nn.conv2d(x, Gx, strides=[1, 1, 1, 1], padding='SAME')
    grad_y = tf.nn.conv2d(x, Gy, strides=[1, 1, 1, 1], padding='SAME')
    
    return tf.sqrt(tf.square(grad_x) + tf.square(grad_y) + 1e-6)

def boundary_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_true.shape.rank == 3:
        y_true = tf.expand_dims(y_true, -1)
    if y_pred.shape.rank == 3:
        y_pred = tf.expand_dims(y_pred, -1)

    edge_true = sobel_edges(y_true)
    edge_pred = sobel_edges(y_pred)

    return tf.reduce_mean(tf.square(edge_true - edge_pred))

#may remove
def combined_dice_boundary_loss(alpha=0.5, beta=0.5):
    def loss(y_true, y_pred):
        y_pred_probs = tf.sigmoid(y_pred)  # Convert logits to probabilities
        dice = -tf.math.log(dice_coef(y_true, y_pred_probs) + 1e-6)
        b_loss = boundary_loss(y_true, y_pred_probs)
        return alpha * dice + beta * b_loss
    return loss


