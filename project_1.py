from create_panorama import *
from plotting import *
from quaternion_operations import *
from preprocess_data import *
from load_data import *
from constants import *

from tqdm import tqdm
import time

import numpy as np
import transforms3d
from PIL import Image

import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


def cost_fun(q):
    '''
    Cost function based on the formula to calculate c from the PR1 PDF file
    '''
    s1, s2 = 0, 0
    s1 = 0.5 * jnp.power(jnp.linalg.norm(2 * quat_log_v(quat_multi_v(
        quat_inverse_v(q[:, 1:]), quat_multi_v(q, Exp)[:, :-1]))), 2)
    s2 = 0.5 * jnp.linalg.norm(A_corrected[:, 1:] - calculate_a(q)[1:])**2
    S1_loss.append(s1.astype(jnp.float64))
    S2_loss.append(s2.astype(jnp.int32))
    return s1 + s2


def orientation_from_W(W, Q, T):
    '''
    Calculating all orientation values sequentially. Also precomputing the value of exponential function.
    '''
    Exp = quat_exp_v(jnp.vstack((jnp.zeros(W.shape[1]-1), T*W[:, :-1]/2)))
    for i in range(W.shape[1]-1):
        Q = Q.at[:, i+1].set(quat_multi(Q[:, i], Exp[:, i]))
    return Exp, Q


def run_grad(Q, cost_fun, iterations=100, step_size=0.05):
    '''
    Running the gradient descent
    '''
    Q_iters = []
    Q_iters.append(Q[:, 1:])
    start = time.time()
    for iter in tqdm(range(iterations)):
        C = jax.grad(cost_fun)(Q_iters[-1] + perturb)
        Q_iters.append(Q_iters[-1]-(step_size)*C)
    print(time.now()-start)
    return Q_iters


def create_vicd(timestamps, R):
    '''
    Create a data structure similar to the VICON data provided in the train dataset
    '''
    vicd = {'ts': timestamps, 'rots': np.zeros((3, 3, R.shape[1]))}
    for i, r in enumerate(R.T):
        vicd['rots'][:, :, i] = transforms3d.euler.euler2mat(r[0], r[1], r[2])
    return vicd

# Functions for panorama construction


def map_timestamps(vicd, camd):
    '''
    Map the closest timestamps in camd to vicd and remove unrequired VICON rotation matrices
    '''
    cam_time_map = {}
    for t in camd['ts'][0]:
        temp = np.abs(vicd['ts'][0] - t)
        index = np.argmin(temp)
        cam_time_map[t] = index
    return cam_time_map


dataset_number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
dataset_folder = ['trainset', 'trainset', 'trainset', 'trainset', 'trainset',
                  'trainset', 'trainset', 'trainset', 'trainset', 'testset', 'testset', ]
has_cam = [True, True, False,  False,  False,
           False,  False, True, True, True, True]
has_vic = [True, True, True, True, True, True, True, True, True, False,  False]

if __name__ == '__main__':
    args = sys.argv[1:]
    try:
        fast = bool(args[0]=='True' or args[0]=='true' or args[0]=='fast') if args else False
    except:
        fast = False
    while True:
        folder_path = input(
            'Enter root folder path where both traindata and testdata folders are located:' or "./")
        i = int(input('Enter dataset to run for (1-11):')) - 1
        if i < 0 or i > 11:
            print('Invalid dataset entry. Enter value between 1 and 11.')
            continue
        print('Calculations for dataset {}'.format(dataset_number[i]))
        start = time.time()
        S1_loss, S2_loss = [], []
        
        try:
            camd, imud, vicd = load_data(
                folder_path, dataset_number[i], dataset_folder[i], has_cam[i], has_vic[i])
        except:
            print('Directory incorrect. Checking current folder for datasets')
            try:
                camd, imud, vicd = load_data(
                    folder_path, dataset_number[i], dataset_folder[i], has_cam[i], has_vic[i])
            except:
                print('No datasets found in this directory. Please check the directory.')
                break

        print('Calculating constants and correcting bias')
        scale_factor_gyro = scale_factor(Vref_gyro, 10, sensitivity_gyro)
        scale_factor_acc = scale_factor(Vref_acc, 10, sensitivity_acc)

        imu_biases = calculate_bias(threshold_index+1, imud['vals'])
        imud_new = imud.copy()
        # Correcting data using bias and scale factor
        A_corrected = IMU_value(
            imud['vals'][0:3, :], imu_biases[:3], sensitivity_acc, Vref_acc, 10)
        W_corrected = IMU_value(
            imud['vals'][3:, :], imu_biases[3:], sensitivity_gyro, Vref_gyro, 10)
        W_corrected = W_corrected.at[jnp.where(W_corrected == 0)].set(perturb)
        # Multiplying and adding gravity value to show gravity units
        A_corrected = jnp.multiply(A_corrected * g, jnp.array([-1, -1, 1])[
                                   :, jnp.newaxis]) + jnp.array([0, 0, g])[:, jnp.newaxis]
        # Assigning corrected values to separate variables
        Ax, Ay, Az = A_corrected

        print('Calculating motion model from IMU data')
        # Grouping Ws
        W = jnp.vstack((W_corrected[1], W_corrected[2], W_corrected[0]))
        Q = jnp.zeros((4, W.shape[1]), dtype=np.float64)
        Q = Q.at[0, 0].set(1)
        # Calculation of Exponential and Q
        Exp, Q = orientation_from_W(jnp.vstack(
            (W_corrected[1], W_corrected[2], W_corrected[0])), Q, imud_new['ts'][0][1:] - imud_new['ts'][0][:-1])
        A = calculate_a(Q)

        # Gradient Descent
        print('Started training')
        Q_iters = jnp.array(run_grad(Q, cost_fun, iterations, step_size))
        print('Loss started at: {}'.format(S2_loss[0].astype(np.int32)))
        print('Loss ended at: {}'.format(S2_loss[-1].astype(np.int32)))

        # Plotting the graph
        print('Plotting and saving graphs')
        R = plot_euler_angles(Q, vicd, Q_iters[-1], dataset_number[i])
        plot_angular_acceleration(
            A, A_corrected, calculate_a(Q_iters[-1]), dataset_number[i])
        vicd = create_vicd(imud['ts'][:, :R.shape[1]],
                           R) if not has_vic[i] else vicd
        print('Dataset {} processed in {}'.format(
            dataset_number[i], str(time.now()-start)))

        if has_cam[i]:
            print('Starting construction of panorama')
            image = create_panorama_lambert(vicd, camd, fast)
            Image.fromarray(image.astype(np.uint8)).save(
                'lambert_panorama_{}.jpg'.format(dataset_number[i]))
            print('Lambert image stored')
            image = create_panorama(vicd, camd, fast)
            Image.fromarray(image.astype(np.uint8)).save(
                'panorama_{}.jpg'.format(dataset_number[i]))
            print('Image stored')

        print('Time elapsed: ', time.now()-start)
        break
