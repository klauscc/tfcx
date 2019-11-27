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

import os
import time
import tensorflow as tf
import tensorflow.keras as keras
from easydict import EasyDict

from .callbacks.ckpt_callbacks import CheckpointCallback


class BasicTrainer(object):
    """Basic Trainer. The keras model is good enough."""

    def __init__(self, params, net):
        super(BasicTrainer, self).__init__()
        self.params = params
        self.net = net

        self.print = self.params.get("print_fn", print)

        self.initiate()

        if params.get("debug", False):
            self.train_step = tf.function(self.train_step)
            self.test_step = tf.function(self.test_step)

    def initiate(self):
        #initiate metrics
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.train_metrics = EasyDict({"train_loss": self.train_loss})
        self.val_metrics = EasyDict({"val_loss": self.val_loss})

        self.metrics = EasyDict()
        self.metrics.update(self.train_metrics)
        self.metrics.update(self.val_metrics)
        #initiate logs
        self.logs = EasyDict()
        self.update_logs()
        #initiate optimizer
        self.optimizer = self.get_optimizer()
        self.net.optimizer = self.optimizer
        # checkpoint
        self.define_ckpt()
        #callbacks
        self.callbacks = self.define_default_callbacks()
        for callback in self.callbacks:
            callback.set_model(self.net)

    def loss_object(self, inputs, predictions):
        """calculate the loss between predictions and targets.

        Args:
            inputs: Dict. The output of the dataset. Both the model inputs 
                    and targets are contained in the dict.
            predictions: Dict or tuple or Tensor. The output of the model.

        Returns: Tensor. The loss.

        """
        pass

    def train_step(self, inputs):
        """TODO: Docstring for train_step.

        Args:
            inputs (TODO): TODO

        Returns: TODO

        """
        pass

    def test_step(self, inputs):
        """TODO: Docstring for test_step.

        Args:
            inputs (TODO): TODO

        Returns: TODO

        """
        pass

    def add_metric(self, mode, metric_key, metric):
        """add metric

        Args:
            mode: String. train or test or both.
            metric_key: String. The key of the metric.
            value: tf.keras.metrics. 

        Returns: The metric

        """
        if mode.lower() == "train":
            key = "train_" + metric_key
            self.train_metrics[key] = metric
            self.metrics[key] = metric
        else:
            key = "val_" + metric_key
            self.val_metrics[key] = metric
            self.metrics[key] = metric
        return metric

    def add_callbacks(self, callbacks):
        """add additional callbacks.

        Args:
            callbacks: List. Each element is an tf.keras.callbacks.Callback.

        Returns: None

        """
        for callback in callbacks:
            callback.set_model(self.net)
        self.callbacks += callbacks

    def define_ckpt(self):
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                                        optimizer=self.net.optimizer,
                                        model=self.net)
        return self.ckpt

    def define_default_callbacks(self):
        """define default callbacks: ModelSaveCallbacks
        Returns: TODO

        """
        callbacks = []
        params = self.params
        ckpt_path = os.path.join(params.workspace, "ckpts")
        log_dir = os.path.join(params.workspace, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)

        #checkpoint callback
        callbacks.append(
            CheckpointCallback(ckpt_path,
                               ckpt=self.ckpt,
                               save_freq=params.get("save_freq", 1),
                               max_to_keep=params.get("max_to_keep", 100)))
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
        return callbacks

    def update_logs(self):
        for k, v in self.metrics.items():
            self.logs[k] = v.result()

    def get_optimizer(self):
        """TODO: Docstring for get_optimizer.
        Returns: TODO

        """
        params = self.params
        other_args = {}
        if "clipnorm" in params:
            other_args["clipnorm"] = params.clipnorm
        if "clipvalue" in params:
            other_args["clipvalue"] = params.clipvalue

        if params.lr_decay_policy == "exp":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(params.init_lr,
                                                                params.decay_steps,
                                                                params.decay_rate,
                                                                staircase=params.staircase)
        else:
            lr = params.init_lr
        if params.optimizer == "adam":
            beta_1 = params.get("beta_1", 0.9)
            beta_2 = params.get("beta_2", 0.999)
            optimizer = tf.keras.optimizers.Adam(lr, beta_1=beta_1, beta_2=beta_2, **other_args)
        elif params.optimizer == "sgd":
            momentum = params.get("momentum", 0.9)
            optimizer = tf.keras.optimizers.SGD(lr, momentum=momentum, **other_args)
        else:
            raise ValueError("unsupported optimizer: {}".format(params.optimizer))
        return optimizer

    def summary_net(self, net):
        if hasattr(net, "summary"):
            net.summary(print_fn=self.print)
            for layer in net.layers:
                if hasattr(layer, "summary"):
                    self.summary_net(layer)

    def print_metrics(self, metric_dict):
        res = ""
        for k, v in metric_dict.items():
            res += "{}:{:.7f},".format(k, v.result())
        return res

    def _get_size_from_inputs(self, inputs):
        if isinstance(inputs, (list, tuple)):
            return inputs[0].shape[0]
        elif isinstance(inputs, dict):
            return inputs[list(inputs.keys())[0]].shape[0]
        else:
            raise ValueError("inputs is not in (list, tuple, dict), but :{}".format(type(inputs)))

    def train(self, datasets, test_datasets=None, test_step=None, epochs=400, test_period=1):
        """TODO: Docstring for train.
        Returns: TODO

        """

        begin_time = time.time()
        fmt = "Epoch {}, step {}, cost {:.4f}s. metrics: {}"
        callbacks = self.callbacks
        [metric.reset_states() for k, metric in self.metrics.items()]    #reset metrics
        [callback.on_train_begin() for callback in callbacks]
        for epoch in range(epochs):
            #train
            [callback.on_epoch_begin(epoch, self.logs) for callback in callbacks]
            self.print("epoch {}. learning rate: {}".format(
                epoch,
                self.net.optimizer._decayed_lr(tf.float32).numpy()))
            epoch_t1 = time.time()
            for step, inputs in enumerate(datasets):

                t1 = time.time()

                logs = self.logs.copy()
                logs["size"] = self._get_size_from_inputs(inputs)
                logs["batch"] = step
                [callback.on_train_batch_begin(step, logs) for callback in callbacks]

                self.train_step(inputs)
                self.update_logs()

                logs = self.logs.copy()
                logs["size"] = self._get_size_from_inputs(inputs)
                logs["batch"] = step
                [callback.on_train_batch_end(step, logs) for callback in callbacks]

                if epoch == 0 and step == 0:
                    self.summary_net(self.net)
                t2 = time.time()
                self.print(fmt.format(epoch, step, t2 - t1, self.print_metrics(self.train_metrics)))

            #test
            if test_datasets and (epoch + 1) % test_period == 0:
                self.print("begin evaluation. epoch {}".format(epoch))
                [callback.on_test_begin(self.logs) for callback in callbacks]
                for step, inputs in enumerate(test_datasets):
                    if test_step and step == test_step:
                        break
                    t1 = time.time()

                    logs = self.logs.copy()
                    logs["size"] = self._get_size_from_inputs(inputs)
                    logs["batch"] = step
                    [callback.on_test_batch_begin(step, logs) for callback in callbacks]

                    result = self.test_step(inputs)
                    self.update_logs()

                    logs = self.logs.copy()
                    logs["size"] = self._get_size_from_inputs(inputs)
                    logs["batch"] = step
                    logs["result"] = result
                    [callback.on_test_batch_end(step, logs) for callback in callbacks]

                    t2 = time.time()
                    self.print(
                        fmt.format(epoch, step, t2 - t1, self.print_metrics(self.val_metrics)))
                [callback.on_test_end(self.logs) for callback in callbacks]

            [callback.on_epoch_end(epoch, self.logs) for callback in callbacks]
            for k, metric in self.metrics.items():
                metric.reset_states()

            epoch_t2 = time.time()
            self.print("Epoch {}. Total Time: {:.4f}s".format(epoch, epoch_t2 - epoch_t1))
        end_time = time.time()
        self.print("Training cost total:{:.2f}s".format(end_time - begin_time))

    def test(self, test_dataset, ckpt_path):
        """TODO: Docstring for test.

        Args:
            test_dataset: The dataset to test.

        Returns: TODO

        """
        fmt = "Test step {}, cost {:.4f}s. metrics: {}"
        self.print("Begin Test for {}".format(ckpt_path))
        self.print("load weight from : {}".format(ckpt_path))

        for inputs in test_dataset.take(1):
            self.test_step(inputs)
            self.ckpt.restore(ckpt_path)

        [metric.reset_states() for k, metric in self.metrics.items()]
        epoch = os.path.basename(ckpt_path)[8:11]
        for callback in self.callbacks:
            callback.epoch = epoch

        for step, inputs in enumerate(test_dataset):
            t1 = time.time()
            result = self.test_step(inputs)
            logs = self.logs.copy()
            logs["result"] = result
            [callback.on_test_batch_end(step, logs) for callback in self.callbacks]
            t2 = time.time()
            self.print(fmt.format(step, t2 - t1, self.print_metrics(self.val_metrics)))
        self.print("Test Done.")
