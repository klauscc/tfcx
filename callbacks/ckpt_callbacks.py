# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2019/09/26
#   description:
#
#================================================================

import tensorflow as tf


class CheckpointCallback(tf.keras.callbacks.Callback):
    """save and load tf checkpoints"""

    def __init__(self, filepath, ckpt=None, save_freq=1, max_to_keep=100):
        super(CheckpointCallback, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq
        self.max_to_keep = max_to_keep
        self.ckpt = ckpt
        self.ckpt_manager = None

    def set_ckpt_manager(self):
        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                            optimizer=self.model.optimizer,
                                            model=self.model)
        if self.ckpt_manager is None:
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt,
                                                           self.filepath,
                                                           max_to_keep=self.max_to_keep)
        return self.ckpt_manager

    def on_train_batch_end(self, batch, logs):
        if self.ckpt is None:
            self.set_ckpt_manager()
        self.ckpt.step.assign_add(1)

    def on_epoch_end(self, epoch, logs):
        if (epoch + 1) % self.save_freq == 0:
            manager = self.set_ckpt_manager()
            save_path = manager.save(checkpoint_number=epoch)
            print("Save checkpoint for epoch {}/ step {} : {}".format(epoch, int(self.ckpt.step),
                                                                      save_path))
