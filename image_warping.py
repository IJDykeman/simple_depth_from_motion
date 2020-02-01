import tensorflow as tf
import tf_lie


def warp(pixel_coords, depth_image, se3_u, se3_w):

    # equation 3 in engel 2014
    # https://vision.in.tum.de/_media/spezial/bib/engel14eccv.pdf
    l = [pixel_coords[...,1] / depth_image,
         pixel_coords[...,0] / depth_image,
         1.0 / depth_image,
         tf.ones_like(depth_image)]
    homogenous_pixel_coords = tf.expand_dims((tf.stack(l, axis=-1)), axis=-1)
    transform = tf_lie.SE3_from_uw(se3_u, se3_w)

    # blank_transform_map is an all zeros tensor if shape
    # [batch size, height, width, 4, 4] that will contain the 4x4
    # homogenous transform for each pixel
    blank_transform_map = tf.zeros(homogenous_pixel_coords.shape[:3].as_list() + [4,4])

    # change the transform matrix batch's shape so we can broadcast
    # it over the spatial dimensions of the image batch
    transform = tf.reshape(transform, [-1,1,1,4,4])
    camera_transform_map = transform + blank_transform_map
   
    n_splits = 4 #keep points per split 40000 or lower
    camera_transform_map_parts = tf.split(camera_transform_map,num_or_size_splits=n_splits,axis=1)
    homogenous_pixel_coords_parts = tf.split(homogenous_pixel_coords,num_or_size_splits=n_splits,axis=1) 
    
    parts_list = []
    for ctm,hpc in zip(camera_transform_map_parts,homogenous_pixel_coords_parts):
        parts_list.append(tf.matmul(ctm,hpc)[...,0])
    warped_pixel_location = tf.concat(parts_list,1) 
    
    return tf.stack([warped_pixel_location[...,0], 
                     warped_pixel_location[...,1],
                     tf.ones_like(depth_image)], axis=-1) \
                        / tf.expand_dims(warped_pixel_location[...,2], axis=-1)

def warp_image(image_batch, depth_image, se3_u, se3_w):
    
    n_rows = image_batch.shape.as_list()[1]
    n_cols = image_batch.shape.as_list()[2]
    rows = tf.range(0.0, n_rows, 1.0) / n_rows
    cols = tf.range(0.0, n_cols, 1.0) / n_cols
    coords = tf.stack(tf.meshgrid(cols, rows), axis=-1)
    
    warped_normalized_pixel_coords = warp(coords, depth_image, se3_u, se3_w)[...,:2]
    warped_pixel_coords  = warped_normalized_pixel_coords * tf.Variable([n_rows * 1.0,
                                                                         n_cols * 1.0])
    
    warped_image = tf.contrib.resampler.resampler(
        image_batch,
        warped_pixel_coords[...,::-1])
    
    valid_coord_mask = tf.cast(warped_pixel_coords[...,0] > 0, tf.float32) * \
                       tf.cast(warped_pixel_coords[...,0] < n_rows-1, tf.float32) * \
                       tf.cast(warped_pixel_coords[...,1] > 0, tf.float32)* \
                       tf.cast(warped_pixel_coords[...,1] < n_cols-1, tf.float32)
    warped_image *= tf.expand_dims(valid_coord_mask, axis=-1)
    return warped_image
