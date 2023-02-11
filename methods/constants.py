import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)
'''Constant data from datasheet'''
Vref_acc = 3.3*1000 # in mV
Vref_gyro = 3.3*1000  # in mV (not same as the sheet)
sensitivity_acc = 300
sensitivity_gyro = 3.33*180/jnp.pi
g = 9.81
perturb = 0.001

'''Miscellaneous constants'''
threshold_index = 400
zero_g = jnp.array([0., 0., 0., g])
iterations = 300
step_size = 0.025
folder_path = "./"
vert, hor = 45., 60.