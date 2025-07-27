from typing import Dict
from dataclasses import dataclass

val_dtype = "float32"
thread_extent_for_short = 128
thread_extent_for_long = 128
length_threshold = 32
long_align_val = 16
short_align_val = 8


@dataclass
class DeviceConfig:
    thread_consume_element_cycle: int
    shared_consume_element_cycle: int
    l2_store_latency: int

device_config_map: Dict[str, DeviceConfig] = {}

def register_device(device_name, config: DeviceConfig):
    device_config_map[device_name] = config
    return config

register_device("hopper", DeviceConfig(53, 4, 150))
register_device("ampere", DeviceConfig(53, 4, 300))
register_device("iluvatar", DeviceConfig(53, 6, 400))


def update_config(short_align_val_, thread_extent_for_short_, length_threshold_, long_align_val_, thread_extent_for_long_):
    global thread_extent_for_short, thread_extent_for_long, length_threshold, long_align_val, short_align_val
    thread_extent_for_short = thread_extent_for_short_
    thread_extent_for_long = thread_extent_for_long_
    length_threshold = length_threshold_
    long_align_val = long_align_val_
    short_align_val = short_align_val_

def print_config():
    print("-------------used config-------------")
    print("thread_extent_for_short: ", thread_extent_for_short)
    print("thread_extent_for_long: ", thread_extent_for_long)
    print("length_threshold: ", length_threshold)
    print("long_align_val: ", long_align_val)
    print("short_align_val: ", short_align_val)