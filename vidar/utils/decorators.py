# Copyright 2023 Toyota Research Institute.  All rights reserved.

from knk_vision.vidar.vidar.utils.types import is_seq, is_dict


def iterate1(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var, *args, **kwargs):
        if is_seq(var):
            return [func(v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(var, *args, **kwargs)
    return inner


def iterate2(func):
    """Decorator to iterate over a list (second argument)"""
    def inner(self, var, *args, **kwargs):
        if is_seq(var):
            return [func(self, v, *args, **kwargs) for v in var]
        elif is_dict(var):
            return {key: func(self, val, *args, **kwargs) for key, val in var.items()}
        else:
            return func(self, var, *args, **kwargs)
    return inner


def iterate12(func):
    """Decorator to iterate over a list (first argument)"""
    def inner(var1, var2, *args, **kwargs):
        if is_seq(var1) and is_seq(var2):
            return [func(v1, v2, *args, **kwargs) for v1, v2 in zip(var1, var2)]
        elif is_dict(var1) and is_dict(var2):
            return {key: func(val1, val2, *args, **kwargs)
                    for key, val1, val2 in zip(var1.keys(), var1.values(), var2.values())}
        else:
            return func(var1, var2, *args, **kwargs)
    return inner


def multi_write(func):
    """Decorator to write multiple files"""
    def inner(filename, data, **kwargs):
        if is_seq(data):
            for i in range(len(data)):
                filename_i, ext = filename.split('.')
                filename_i = '%s_%d.%s' % (filename_i, i, ext)
                func(filename_i, data[i], **kwargs)
            return
        elif is_dict(data):
            for key, val in data.items():
                filename_i, ext = filename.split('.')
                filename_i = '%s(%s).%s' % (filename_i, key, ext)
                func(filename_i, val, **kwargs)
            return
        else:
            return func(filename, data)
    return inner
