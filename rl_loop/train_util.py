"""
@file: rl_training.py
Created on 23.06.22
@project: CrazyAra
@author: queensgambit

Utility methods for training with one of the training agents.
"""
import rl_loop.metrics_pytorch as pytorch_metrics


def get_metrics(train_config):
    """
    Returns the metrics according to the used training framework.
    :param train_config: Training configuration object
    :return: Training metrics
    """
    if train_config.framework == 'pytorch':
        return _get_pytorch_metrics(train_config)

def _get_pytorch_metrics(train_config):
    metrics_pytorch = {
        'value_loss': pytorch_metrics.MSE(),
        'policy_loss': pytorch_metrics.CrossEntropy(train_config.sparse_policy_label),
        'value_acc_sign': pytorch_metrics.AccuracySign(),
        'policy_acc': pytorch_metrics.Accuracy(train_config.sparse_policy_label)
    }
    if train_config.use_wdl:
        metrics_pytorch['wdl_loss'] = pytorch_metrics.CrossEntropy(True)
        metrics_pytorch['wdl_acc'] = pytorch_metrics.Accuracy(True)
    if train_config.use_plys_to_end:
        metrics_pytorch['plys_to_end_loss'] = pytorch_metrics.MSE()

    return metrics_pytorch
