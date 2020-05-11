class Network:
    _init = None
    session = None
    optimizer = None
    saver = None
    losses = None
    x = None
    y = None
    predictions = None
    probabilities = None
    network_output = None
    local_variables = None
    global_variables = None
    training = None
    learning_rate = None
    def __init__(self, learning_rate, training, optimizer, losses, x,  y,  network_output, local_variables, global_variables):
        import tensorflow as tf
        from constants import Constants
        self.tf=tf
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=tf.get_default_graph(),config=config)
        self.network_output = network_output
        self.session.run(local_variables)
        self.session.run(global_variables)
        self.optimizer = optimizer
        self.training = training
        self.learning_rate = learning_rate
        self.saver = tf.train.Saver(max_to_keep=Constants.saved_models_count)
        self.losses = losses
        self.x = x
        self.y = y
        self.probabilities = {name: val['out_soft'] for name, val in network_output.items()}
        self.local_variables = local_variables
        self.global_variables = global_variables

    def clear_graph(self):
        self.tf.reset_default_graph()
        self.session.run(self.local_variables)
        self.session.run(self.global_variables)

    def reset_local_variables(self):
        self.session.run(self.local_variables)
