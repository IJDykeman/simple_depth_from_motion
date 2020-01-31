# simple_depth_from_motion
uses tensorflow gradient to find a depth map to fit the images
But some tensorflow GPU versions fail. 
I think, because the tf.matmul operation in these versions can not
handle large batches. So I split up the large matmul uperation into
10 chunks, and now it runs on my gpu.
