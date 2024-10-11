from __future__ import print_function
import os


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def print_log(out_f, message):
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_train_log(epoch, iteration, errors, t):
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def format_validation_log(epoch, iteration, errors, t):
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    for k, v in errors.items():
        message += '%s: %.3f ' % (k, v)
    return message


def listdir_full_path(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

