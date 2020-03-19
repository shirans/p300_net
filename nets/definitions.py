import torch.nn as nn

def conv2d_output_shape(input_hight, input_width, kenel_height, kernel_width, padding, stride):
    return (input_hight - kenel_height + 2 * padding) / stride + 1, \
           (input_width - kernel_width + 2 * padding) / stride + 1,


class Conv2dShapes():
    def __init__(self, input_channels, num_filters, channels, timestamps, stride, padding, filter_height, filter_width):
        self.input_channels = input_channels
        self.padding = padding  # P
        self.stride = stride  # S
        self.channels = channels  # H
        self.timestamps = timestamps  # W
        self.num_filters = num_filters  # K
        self.filter_height = filter_height  # F_W
        self.filter_width = filter_width  # F_H

    def create_conv2d(self):
        return nn.Conv2d(in_channels=self.input_channels,
                         out_channels=self.num_filters,
                         kernel_size=(self.filter_height, self.filter_width),
                         stride=self.stride, padding=self.padding)