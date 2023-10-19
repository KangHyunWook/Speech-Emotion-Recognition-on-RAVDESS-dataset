import tensorflow as tf

def get_config():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('batch_size', 128, 'the batch size for a training iteration.')
    flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
    flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
    flags.DEFINE_integer('coarsening_levels', 0, 'Number of coarsened graphs.')

    return FLAGS
