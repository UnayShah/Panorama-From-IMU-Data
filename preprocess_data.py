import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

'''Data operations to preprocess IMU and VICON data'''


def calculate_bias(end_index: int, data_row) -> int:
    '''
    Taking average of all values till a particular index to calculate the bias
    '''
    return jnp.average(data_row[:, :end_index], axis=1)

# remove bias from data by subtracting that number


def remove_bias(biases, data):
    '''
    Subtract the bias term from the data
    '''
    return data - biases[:, jnp.newaxis]


def scale_factor(Vref, bit_count, sensitivity_in_rad):
    '''
    Calculate scale factor of the sensor
    '''
    return (Vref/(2**bit_count))/sensitivity_in_rad


def IMU_value(value, bias, sensitivity, Vref, bit_count):
    '''
    Calculate the value of the IMU sensor using formula from the datasheet
    '''
    return (value-bias[:, jnp.newaxis]) * scale_factor(Vref, bit_count, sensitivity)
