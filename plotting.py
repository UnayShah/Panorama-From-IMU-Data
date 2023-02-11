import matplotlib as plt

import transforms3d
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)
'''Functions to plot results'''


def plot_euler_angles(Q, vicd, trained_Q, prefix):
    '''
    Plot the euler angles to compare vicon data with predicted values and trained values.
    If vicon data is not available, then it is not plotted.
    Saves the plots as images names prefix_euler.jpg
    '''
    Q_to_V = jnp.zeros((3, Q.shape[1]))
    trained_Q_to_V = jnp.zeros((3, trained_Q[0, :].shape[0]))

    for v in range(Q.shape[1]):
        Q_to_V = Q_to_V.at[:, v].set(
            jnp.array(transforms3d.euler.quat2euler(Q[:, v])))
        trained_Q_to_V = trained_Q_to_V.at[:, v].set(
            jnp.array(transforms3d.euler.quat2euler(trained_Q[:, v])))

    if vicd:
        V = jnp.zeros((3, vicd['rots'].shape[2])) if vicd else None
        for v in range(vicd['rots'].shape[2]):
            V = V.at[:, v].set(jnp.array(transforms3d.euler.mat2euler(
                vicd['rots'][:, :, v]))) if vicd else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Dataset {} Euler Angles'.format(prefix))
    titles = ['Euler angle X', 'Euler angle Y', 'Euler angle Z']
    for i in range(3):
        axes[i].set_title(titles[i])
        axes[i].plot(Q_to_V[i, :])
        axes[i].plot(trained_Q_to_V[i, :])
        if vicd:
            axes[i].plot(V[i, :])
            axes[i].legend(['Predicted Values from W',
                           "Trained Motion model", "Values from VICON"])
        else:
            axes[i].legend(['Predicted Values from W',
                           "Trained Motion model", ])
    plt.savefig('{}_W.jpg'.format(prefix))
    plt.close()
    return trained_Q_to_V


def plot_angular_acceleration(A, A_corrected, trained_A, prefix):
    '''
    Plot the acceleration to compare IMU data with predicted values and trained values.
    Saves the plots as images names prefix_acc.jpg
    '''
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Dataset {} Acceleration'.format(prefix))
    titles = ['Acceleration X', 'Acceleration Y', 'Acceleration Z']
    for i in range(3):
        axes[i].set_title(titles[i])
        axes[i].plot(A[i+1, :])
        axes[i].plot(trained_A[i+1, :])
        axes[i].plot(A_corrected[i])
        axes[i].legend(["Predicted values from Q",
                       "Trained Observation model", 'Values from IMU', ])
    plt.savefig('{}_A.jpg'.format(prefix))
    plt.close()


def plot_cost(S2_loss, dataset_number):
    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    axes.set_title('Loss for dataset {}'.format(dataset_number))
    axes.plot(S2_loss)
    plt.savefig('{}_cost.jpg'.format(dataset_number))
    plt.close()
