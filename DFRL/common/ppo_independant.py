import tensorflow as tf
import numpy as np

class ValueNetwork():
    def __init__(self, num_features, hidden_size, learning_rate=.01):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.tf_graph = tf.Graph()
        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, self.num_features], dtype=tf.float32)
            self.W = [
                tf.compat.v1.get_variable("W1", shape=[self.num_features, self.hidden_size]),
                tf.compat.v1.get_variable("W2", shape=[self.hidden_size, self.hidden_size]),
                tf.compat.v1.get_variable("W3", shape=[self.hidden_size, 1])
            ]
            self.B = [
                tf.compat.v1.get_variable("B1", [self.hidden_size]),
                tf.compat.v1.get_variable("B2", [self.hidden_size]),
                tf.compat.v1.get_variable("B3", [1]),
            ]
            self.layer_1 = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.layer_2 = tf.nn.relu(tf.matmul(self.layer_1, self.W[1]) + self.B[1])
            self.output = tf.reshape(tf.matmul(self.layer_2, self.W[2]) + self.B[2], [-1])

            self.rollout = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)
            self.loss = tf.losses.mean_squared_error(self.output, self.rollout)
            self.grad_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.minimize = self.grad_optimizer.minimize(self.loss)

            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    def get(self, states):
        value = self.session.run(self.output, feed_dict={self.observations: states})
        return value

    def update(self, states, discounted_rewards):
        _, loss = self.session.run([self.minimize, self.loss], feed_dict={
            self.observations: states, self.rollout: discounted_rewards
        })


class PPOPolicyNetwork():
    def __init__(self, num_features, layer_size, num_actions, epsilon=.1,
                 learning_rate=9e-4):
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.compat.v1.get_variable("W1", shape=[num_features, layer_size]),
                tf.compat.v1.get_variable("W2", shape=[layer_size, layer_size]),
                tf.compat.v1.get_variable("W3", shape=[layer_size, num_actions])
            ]
            self.B = [
                tf.compat.v1.get_variable("B1", [layer_size]),
                tf.compat.v1.get_variable("B2", [layer_size]),
                tf.compat.v1.get_variable("B3", [num_actions]),
            ]
            trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.saver = tf.compat.v1.train.Saver(trainable_vars, max_to_keep=3000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            self.output = tf.nn.softmax(tf.matmul(self.output, self.W[2]) + self.B[2])

            self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / (self.old_responsible_outputs + + 1e-10)

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            ) - 0.03 * self.new_responsible_outputs * tf.math.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.B0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad, self.B0_grad, self.B1_grad,
                                          self.B2_grad]
            self.trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states):
        dist = self.session.run(self.output, feed_dict={self.observations: states})
        return dist

    def update(self, states, chosen_actions, ep_advantages):
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states})
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],
            self.B0_grad: self.gradients[3][0],
            self.B1_grad: self.gradients[4][0],
            self.B2_grad: self.gradients[5][0],
        })

        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities
        })

    def save_w(self, name):
        with self.tf_graph.as_default():
            self.saver.save(self.session, name)

    def restore_w(self, name):
        with self.tf_graph.as_default():
            self.saver.restore(self.session, name)

class CPPOPolicyNetwork():
    def __init__(self, num_features, layer_size, num_actions, std, if_tanh=False, epsilon=.1, learning_rate=9e-4):
        self.tf_graph = tf.Graph()
        self.std = std
        self.if_tanh = if_tanh

        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)
            self.W = [
                tf.compat.v1.get_variable("W1", shape=[num_features, layer_size]),
                tf.compat.v1.get_variable("W2", shape=[layer_size, layer_size]),
                tf.compat.v1.get_variable("W3", shape=[layer_size, num_actions])
            ]
            self.B = [
                tf.compat.v1.get_variable("B1", [layer_size]),
                tf.compat.v1.get_variable("B2", [layer_size]),
                tf.compat.v1.get_variable("B3", [num_actions]),
            ]
            trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.saver = tf.compat.v1.train.Saver(trainable_vars, max_to_keep=3000)
            if self.if_tanh:
                self.output = tf.nn.tanh(tf.matmul(self.observations, self.W[0]) + self.B[0])
                self.output = tf.nn.tanh(tf.matmul(self.output, self.W[1]) + self.B[1])
            else:
                self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
                self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            self.output = tf.nn.sigmoid(tf.matmul(self.output, self.W[2]) + self.B[2])
            self.mean = self.output

            self.sampling = self.mean + self.std * tf.random_normal(tf.shape(self.mean))
            self.neglogprob = 0.5 * tf.square((self.sampling - self.mean) / self.std) + 0.5 * np.log(2.0 * np.pi) + tf.math.log(self.std)

            self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.OLDNEGLOGPAC = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

            self.old_responsible_outputs = 0.5 * tf.reduce_sum(tf.square((self.chosen_actions - self.mean) / self.std), axis=-1) + 0.5 * np.log(2.0 * np.pi) + tf.math.log(self.std)

            #self.ratio = self.new_responsible_outputs / self.old_responsible_outputs
            self.ratio = tf.exp(self.OLDNEGLOGPAC - self.old_responsible_outputs)

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            )
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.B0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad, self.B0_grad, self.B1_grad,
                                          self.B2_grad]
            self.trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states):
        #sampling, neglogprob, mean = self.session.run([self.sampling, self.neglogprob, self.mean], feed_dict={self.observations: states})
        sampling, neglogprob = self.session.run([self.sampling, self.neglogprob], feed_dict={self.observations: states})
        sampling = np.clip(sampling, 0, 1)
        return sampling, neglogprob

    def get_mean(self, states):
        mean = self.session.run(self.mean, feed_dict={self.observations: states})
        return mean

    def update(self, states, chosen_actions, neglogproba, ep_advantages):
        #is the next line mandatory?
        # self.session.run(self.output, feed_dict={self.observations: states})
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],
            self.B0_grad: self.gradients[3][0],
            self.B1_grad: self.gradients[4][0],
            self.B2_grad: self.gradients[5][0],
        })

        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.OLDNEGLOGPAC: neglogproba[:, 0]
        })

    def save_w(self, name):
        with self.tf_graph.as_default():
            self.saver.save(self.session, name)

    def restore_w(self, name):
        with self.tf_graph.as_default():
            self.saver.restore(self.session, name)


class PPOPolicyNetworkMasked():
    def __init__(self, num_features, layer_size, num_actions, epsilon=.1,
                 learning_rate=9e-4):
        self.tf_graph = tf.Graph()

        with self.tf_graph.as_default():
            self.session = tf.compat.v1.Session()

            self.observations = tf.compat.v1.placeholder(shape=[None, num_features], dtype=tf.float32)

            # Action mask placeholder for variable number of states
            self.action_mask = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32, name='action_mask')

            self.W = [
                tf.compat.v1.get_variable("W1", shape=[num_features, layer_size]),
                tf.compat.v1.get_variable("W2", shape=[layer_size, layer_size]),
                tf.compat.v1.get_variable("W3", shape=[layer_size, num_actions])
            ]
            self.B = [
                tf.compat.v1.get_variable("B1", [layer_size]),
                tf.compat.v1.get_variable("B2", [layer_size]),
                tf.compat.v1.get_variable("B3", [num_actions]),
            ]
            trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.saver = tf.compat.v1.train.Saver(trainable_vars, max_to_keep=3000)

            self.output = tf.nn.relu(tf.matmul(self.observations, self.W[0]) + self.B[0])
            self.output = tf.nn.relu(tf.matmul(self.output, self.W[1]) + self.B[1])
            
            # Compute logits before applying mask
            logits = tf.matmul(self.output, self.W[2]) + self.B[2]
            
            # mask logits by setting them to a large negative value for invalid actions
            masked_logits = logits + (1 - self.action_mask) * (-1e9) 
            
            # Use softmax on masked logits
            self.output = tf.nn.softmax(masked_logits)

            self.advantages = tf.compat.v1.placeholder(shape=[None], dtype=tf.float32)

            self.chosen_actions = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)
            self.old_probabilities = tf.compat.v1.placeholder(shape=[None, num_actions], dtype=tf.float32)

            self.new_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.output, axis=1)
            self.old_responsible_outputs = tf.reduce_sum(self.chosen_actions * self.old_probabilities, axis=1)

            self.ratio = self.new_responsible_outputs / (self.old_responsible_outputs + 1e-10)

            self.loss = tf.reshape(
                tf.minimum(
                    tf.multiply(self.ratio, self.advantages),
                    tf.multiply(tf.clip_by_value(self.ratio, 1 - epsilon, 1 + epsilon), self.advantages)),
                [-1]
            ) - 0.03 * self.new_responsible_outputs * tf.math.log(self.new_responsible_outputs + 1e-10)
            self.loss = -tf.reduce_mean(self.loss)

            self.W0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.W2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.B0_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B1_grad = tf.compat.v1.placeholder(dtype=tf.float32)
            self.B2_grad = tf.compat.v1.placeholder(dtype=tf.float32)

            self.gradient_placeholders = [self.W0_grad, self.W1_grad, self.W2_grad, self.B0_grad, self.B1_grad,
                                          self.B2_grad]
            self.trainable_vars = [item for sublist in [self.W, self.B] for item in sublist]
            self.gradients = [(np.zeros(var.get_shape()), var) for var in self.trainable_vars]

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.get_grad = self.optimizer.compute_gradients(self.loss, self.trainable_vars)
            self.apply_grad = self.optimizer.apply_gradients(zip(self.gradient_placeholders, self.trainable_vars))
            init = tf.compat.v1.global_variables_initializer()
            self.session.run(init)

    def get_dist(self, states, mask=None):
        # Create a default mask of ones if mask is not provided
        if mask is None:
            mask = np.ones((states.shape[0], self.output.shape[1]), dtype=np.float32)
        
        # Ensure mask matches the number of states provided
        assert mask.shape[0] == states.shape[0], "Mask must match the number of states"
        dist = self.session.run(self.output, feed_dict={self.observations: states, self.action_mask: mask})
        return dist

    def update(self, states, chosen_actions, ep_advantages, mask=None):
        # Create a default mask of ones if mask is not provided
        if mask is None:
            mask = np.ones((states.shape[0], self.output.shape[1]), dtype=np.float32)
        
        # Ensure mask matches the number of states provided
        assert mask.shape[0] == states.shape[0], "Mask must match the number of states"
        
        old_probabilities = self.session.run(self.output, feed_dict={self.observations: states, self.action_mask: mask})
        
        # Apply gradients using placeholders for manual control
        self.session.run(self.apply_grad, feed_dict={
            self.W0_grad: self.gradients[0][0],
            self.W1_grad: self.gradients[1][0],
            self.W2_grad: self.gradients[2][0],
            self.B0_grad: self.gradients[3][0],
            self.B1_grad: self.gradients[4][0],
            self.B2_grad: self.gradients[5][0],
        })

        # Recalculate gradients and loss with updated placeholders
        self.gradients, loss = self.session.run([self.get_grad, self.output], feed_dict={
            self.observations: states,
            self.advantages: ep_advantages,
            self.chosen_actions: chosen_actions,
            self.old_probabilities: old_probabilities,
            self.action_mask: mask
        })

    def save_w(self, name):
        with self.tf_graph.as_default():
            self.saver.save(self.session, name)

    def restore_w(self, name):
        with self.tf_graph.as_default():
            self.saver.restore(self.session, name)

