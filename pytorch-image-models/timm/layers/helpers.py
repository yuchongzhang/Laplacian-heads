# Copyright 2020 Ross Wightman
def to_2tuple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value, value)
