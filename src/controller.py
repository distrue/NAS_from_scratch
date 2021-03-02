import tensorflow as tf

from random import seed
from random import random
import time

seed(time.time())

class Controller(object):
  def __init__(
    self
  ):
    self.num_layers = 4
    self.num_branches = 6
    self.lstm_size = 32
    self.num_blocks_per_branch = 6

    self.l2_reg = 1e-4

    self.lstm_weight = []
    for layer_id in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(layer_id)):
        w = tf.get_variable("w", [2 * self.lstm_size, 4 * self.lstm_size])
        self.lstem_weight.append(w)
    
    self.num_configs = (2 ** self.num_blocks_per_branch) - 1

    with tf.variable_scope("embedding"):
      self.embed_graph = tf.get_variable("embed_graph", [1, self.lstm_size])
      self.embed_weight = tf.get_variable("weight", [
        self.num_blocks_per_branch, 
        self.lstm_size
      ])
    
    with tf.variable_scope("softmax"):
      self.softmax_weight = tf.get_variable("weight", [
        self.lstm_size,
        self.num_blocks_per_branch
      ])
    
    with tf.variable_scope("critic"):
      self.critic_weight = tf.get_variable("weight", [self.lstm_size, 1])

    arc_seq = []
    sample_log_probs = []
    all_h = []

    inputs = self.embed_graph
    prev_channel = [
      tf.zeros([1, self.lstm_size], dtype=tf.float32) 
        for _ in range(self.lstm_num_layers)
    ]
    prev_height = [
      tf.zeros([1, self.lstm_size], dtype=tf.float32)
        for _ in range(self.lstm_num_layers)
    ]

    for layer_id in range(self.num_layers):
      for branch_id in range(self.num_branches):
        next_channel, next_height = stack_lstm(
          inputs, 
          prev_channel, 
          prev_height, 
          self.lstm_weight
        )
        all_h.append(tf.stop_gradient(next_height[-1]))

        logits = tf.matmul(next_height[-1], self.softmax_weight)
        logits = 1.10 * tf.tanh(logits)

        config_id = tf.multinomial(logits, 1)
        config_id = tf.to_int32(config_id)
        config_id = tf.reshape(config_id, [1])
        arc_seq.append(config_id)
        log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, 
          labels=config_id
        )

        inputs = tf.nn.embedding_lookup(self.embed_weight, config_id)
    
    self.sample_arc = tf.concat(arc_seq, axis=0)
    
    self.sample_log_probs = tf.concat(sample_log_probs, axis=0)
    self.ppl = tf.exp(
      tf.reduce_sum(self.sample_log_probs) / 
      tf.to_float(self.num_layers, self.num_branches)
    )
    self.all_h = all_h

  def build_trainer(self):
    # TODO: update to child networks reward load
    self.reward = tf.to_float(random())

    self.sample_log_probs = tf.reduce_sum(self.sample_log_probs)
    self.baseline = tf.Variable(0.0, dtype=tf.float32, trainable=False)
    baseline_update = tf.assign_sub(
      self.baseline, 0.001 * (self.baseline - self.reward)
    )
    with tf.control_dependencies([baseline_update]):
      self.reward = tf.identity(self.reward)
    self.loss = self.sample_log_probs * (self.reward - self.baseline)

    self.train_step = tf.Variable(
        0, dtype=tf.int32, trainable=False, name="train_step"
    )

    tf_variables = [
      var for var in tf.trainable_variables()
        if var.name.startswith(self.name)  
          and "critic_weighr" not in var.name
    ]

    print("-" * 80)
    for var in tf_variables:
      print(var)

    if self.l2_reg > 0:
      l2_losses = []
      for var in tf_variables:
        l2_losses.appennd(tf.reduce_sum(var ** 2))
      l2_loss = tf.add_n(l2_losses)
      self.loss += l2_reg * l2_loss

    grads = tf.gradients(self.loss, tf_variables)
    self.grad_norm = tf.global_norm(grads)

    grad_norms = {}
    for v, g in zip(tf_variables, grads):
      if v is None or g is None:
        continue
      if isinstance(g, tf.IndexedSlices):
        grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
      else:
        grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))

    clipped = []
    for g in grads:
      if isinstance(g, tf.IndexedSlices):
        c_g = tf.clip_by_norm(g.values, grad_bound)
        c_g = tf.IndexedSlices(g.indices, c_g)
      else:
        c_g = tf.clip_by_norm(g, grad_bound)
      clipped.append(g)
    grads = clipped

    self.learning_rate = tf.train.exponential_decay(
      lr_init, tf.maximum(train_step - lr_dec_start, 0),
      lr_dec_every,
      lr_dec_rate, staircase=True
    )

    self.optimizer = tf.train.GradientDescentOptimizer(
      learning_rate, use_locking=True)

    self.train_op = optimizer.apply_gradients(
      zip(grads, tf_variables), global_step=train_step)

    
    