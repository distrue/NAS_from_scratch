import tensorflow as tf
import argparse
import os
import sys
sys.path.insert(0, '.')

from .controller import Controller

num_train_examples = 1024
batch_size = 128
num_train_batches = num_train_exmaples / batch_size
train_every = 50

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out_dir',
        default=os.path.join(os.path.dirname(), '..', 'output'),
        help="output result"
    )
    args = parser.parse_args()

    g = tf.Graph()
    with g.as_default():
        saver = tf.train.Saver(max_to_keep=10)
        checkpoint_saver_hook = tf.train.CheckpointSaverHook(
            model_path=args.out_dir,
            saver=saver,
            save_steps=num_train_batches
        )

        hooks = [checkpoint_saver_hook]
        config = tf.ConfigProto(allow_soft_placement=True)

        controller = Controller()
        child_network = ()

        with tf.train.SingularMonitoredSession(
            config=config,
            hooks=hooks,
            checkpoint_dir=args.out_dir
        ) as sess:
            while True:
                for weight_step in range(num_train_batches * train_every):
                    # add latency load process from communication between raspberry pi
                    
                    # update controller's reward with latency / accuracy
                    # g._unsafe_unfinalize()
                    # self.reward = ACC * LAT
                    # g.finalize()

                    step, loss, grad_norm, baseline = sess.run(
                        controller.train_step,
                        controller.loss,
                        controller.grad_norm,
                        controller.baseline
                    )
                

                # for ctrl_step in range(num_ctrl_steps):
                #    sess.run()
                
                # proper network choosing

