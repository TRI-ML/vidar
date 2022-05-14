# TRI-VIDAR - Copyright 2022 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict


def average_key(batch_list, key):
    """
    Average key in a list of batches

    Parameters
    ----------
    batch_list : list[Dict]
        List containing dictionaries with the same keys
    key : String
        Key to be averaged

    Returns
    -------
    average : Float
        Average of the value contained in key for all batches
    """
    values = [batch[key] for batch in batch_list]
    return sum(values) / len(values)


def average_sub_key(batch_list, key, sub_key):
    """
    Average subkey in a dictionary in a list of batches

    Parameters
    ----------
    batch_list : list[Dict]
        List containing dictionaries with the same keys
    key : String
        Key to be averaged
    sub_key : String
        Sub key to be averaged (belonging to key)

    Returns
    -------
    average : Float
        Average of the value contained in the sub_key of key for all batches
    """
    values = [batch[key][sub_key] for batch in batch_list]
    return sum(values) / len(values)


def average_loss_and_metrics(batch_list, prefix):
    """
    Average loss and metrics values in a list of batches

    Parameters
    ----------
    batch_list : list[Dict]
        List containing dictionaries with the same keys
    prefix : String
        Prefix string for metrics logging

    Returns
    -------
    values : Dict
        Dictionary containing a 'loss' float entry and a 'metrics' dict entry
    """
    values = OrderedDict()

    key = 'loss'
    values['{}-{}'.format(prefix, key)] = \
        average_key(batch_list, key)

    key = 'metrics'
    for sub_key in batch_list[0][key].keys():
        values['{}-{}'.format(prefix, sub_key)] = \
            average_sub_key(batch_list, key, sub_key)

    return values

