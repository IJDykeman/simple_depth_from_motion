import tensorflow as tf

def add_two_trailing_dims(x):
    return tf.expand_dims(tf.expand_dims(x, axis=-1), axis=-1)

def transpose_matrix_collection(x):
    axes = list(range(len(x.get_shape())))
    target_axes = axes[:-2] + list(reversed(axes[-2:]))
    return tf.transpose(x, perm=target_axes)

def se3_from_uw(u, w):
    G1 = tf.constant([[0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,0]])
    
    G2 = tf.constant([[0,0,0,0],
                      [0,0,0,1],
                      [0,0,0,0],
                      [0,0,0,0]])
    
    G3 = tf.constant([[0,0,0,0],
                      [0,0,0,0],
                      [0,0,0,1],
                      [0,0,0,0]])
    
    G4 = tf.constant([[0,0, 0,0],
                      [0,0,-1,0],
                      [0,1, 0,0],
                      [0,0, 0,0]])
    
    G5 = tf.constant([[0,0,-1,0],
                      [0,0, 0,0],
                      [1,0, 0,0],
                      [0,0, 0,0]])
    
    G6 = tf.constant([[0,-1, 0,0],
                      [1, 0, 0,0],
                      [0, 0, 0,0],
                      [0, 0, 0,0]])
    translations = tf.cast(tf.stack([G1, G2, G3]), tf.float32)
    rotations =    tf.cast(tf.stack([G4, G5, G6]), tf.float32)
    return tf.reduce_sum(add_two_trailing_dims(u)*translations
                        + add_two_trailing_dims(w)*rotations, axis=-3)


def so3_from_w(w):
    G1 = tf.constant([[0,0, 0],
                      [0,0,-1],
                      [0,1, 0]])
    
    G2 = tf.constant([[0,0,-1],
                      [0,0, 0],
                      [1,0, 0]])
    
    G3 = tf.constant([[0,-1, 0],
                      [1, 0, 0],
                      [0, 0, 0]])
    rotations = tf.cast(tf.stack([G1, G2, G3]), tf.float32)
    return tf.reduce_sum(add_two_trailing_dims(w)*rotations, axis=-3)


def so3_from_SO3(R): # log
    theta = tf.acos((tf.trace(R)-1)/2)
    return add_two_trailing_dims(theta / (2 * tf.sin(theta))) * (R - transpose_matrix_collection(R))

def V_from_R(R):
    wx = so3_from_SO3(R)
    theta = tf.acos((tf.trace(R)-1)/2)
    A = add_two_trailing_dims(tf.sin(theta) / theta)
    B = add_two_trailing_dims((1 - tf.cos(theta)) / (theta ** 2))
    C = (1 - A) / (add_two_trailing_dims(theta) ** 2)
    I = tf.eye(3)
    V = I + B*wx + C*tf.matmul(wx, wx)
    return V

def se3_from_SE3(C): # log
    
    R = C[...,:3,:3]
    t = C[...,:3, 3]
    wx = so3_from_SO3(R)

    V = V_from_R(R)
    Vinv = tf.linalg.inv(V)
    u = tf.matmul(Vinv, tf.expand_dims(t, axis=-1))
    empty_row = tf.zeros(V.shape[:-2].as_list() + [1,4]) 
    WXu = tf.concat([wx, u], axis=-1)
    result = tf.concat([WXu, empty_row], axis=-2)
    return result
    


def SE3_from_uw(u, w): # exp
    wx = so3_from_w(w)
    theta = tf.sqrt(tf.reduce_sum(w * w, axis=-1))
    A = add_two_trailing_dims(tf.sin(theta) / (theta))
    B = add_two_trailing_dims((1 - tf.cos(theta)) / ((theta ** 2)))
    C = (1 - A) / (add_two_trailing_dims(theta) ** 2)
    I = tf.eye(3)
    R = I + A*wx + B*tf.matmul(wx, wx)
    V = I + B*wx + C*tf.matmul(wx, wx)
    Vu = tf.matmul(V, tf.expand_dims(u, axis=-1))
    empty_4x4 = tf.zeros(Vu.shape[:-2].as_list() + [1,4]) 
    row_0001 = empty_4x4 + tf.eye(num_rows=1, num_columns=4)[...,::-1]
    RVu = tf.concat([R, Vu], axis=-1)
    result = tf.concat([RVu, row_0001], axis=-2)
    return result
