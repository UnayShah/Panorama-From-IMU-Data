from datetime import datetime
from tqdm import tqdm
import pickle
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import transforms3d
from PIL import Image

import jax
import jax.numpy as jnp
from jax.config import config; config.update("jax_enable_x64", True)


# Constant data from datasheet
Vref_acc = 3.3*1000 # in mV
Vref_gyro = 3.3*1000  # in mV (not same as the sheet)
sensitivity_acc = 300
sensitivity_gyro = 3.33*180/jnp.pi
g = 9.81
perturb = 0.001

# Finding bias for data correction
threshold_index = 400
zero_g = jnp.array([0., 0., 0., g])
iterations = 300
step_size = 0.025
# folder_path = "/content/drive/MyDrive/Course Materials/2 Winter 23/ECE 276A Sensing & Estimation in Robotics/Projects/ECE276A_PR1/"
folder_path = "./"
vert, hor = 45., 60.



def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

def load_data(dataset, setname, has_cam, has_vic):

  cfile = folder_path + setname + "/cam/cam" + dataset + ".p"
  ifile = folder_path + setname + "/imu/imuRaw" + dataset + ".p"
  vfile = folder_path + setname + "/vicon/viconRot" + dataset + ".p"

  ts = tic()
  camd = read_data(cfile) if has_cam else None
  imud = read_data(ifile)
  vicd = read_data(vfile) if has_vic else None
  toc(ts,"Data import")
  return camd, imud, vicd

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



# Calculations for quaternions
@jax.jit
def quat_multi(q, p):
  '''
  Multiply 2 quaternions
  '''
  qs, ps, qv, pv  = q[0], p[0], q[1:], p[1:]
  return jnp.hstack(((qs*ps) - jnp.dot(qv.T, pv), (qs*pv) + (ps*qv) + jnp.cross(qv, pv)))

@jax.jit
def quat_exp_v(q):
  '''
  Vectorized exponential of a quaternion
  '''
  qs, qv = q[0, :], q[1:,:]
  return jnp.exp(qs)*jnp.vstack((jnp.cos(jnp.linalg.norm(qv,axis=0)),(qv/jnp.linalg.norm(qv,axis=0))*jnp.sin(jnp.linalg.norm(qv,axis=0))))

# Calculations for accelaration
@jax.jit
def quat_inverse_v(q):
  '''
  Vectorized inverse calculation for a quaternion
  '''
  return (q*jnp.array([1., -1., -1., -1.])[:, jnp.newaxis])/(jnp.linalg.norm(q, axis=0)**2)

@jax.jit
def quat_multi_v(q, p):
  '''
  Vectorized multiplication for quaternions
  '''
  qs, ps, qv, pv  = q[0, :], p[0, :], q[1:, :], p[1:, :]
  return jnp.vstack(((qs * ps) - jnp.sum(qv * pv, axis=0), (qs * pv) + (ps * qv) + jnp.cross(qv, pv, axis=0)))

@jax.jit
def calculate_a(q):
  '''
  Vectorized calculatation of acceleration given all the quaternions 
  '''
  return quat_multi_v(quat_inverse_v(q), quat_multi_v(zero_g[:, jnp.newaxis], q))

# Functions for cost calculations
@jax.jit
def quat_log_v(q):
  '''
  Vectorized log for quaternions
  '''
  qs, qv = q[0, :], q[1:, :]
  return jnp.vstack((jnp.log(jnp.linalg.norm(q, axis=0)), (qv/jnp.linalg.norm(qv, axis=0))*jnp.arccos(qs/jnp.linalg.norm(q, axis=0))))

def cost_fun(q):
  '''
  Cost function based on the formula to calculate c from the PR1 PDF file
  '''
  s1, s2 = 0, 0
  s1 = 0.5 * jnp.power(jnp.linalg.norm(2 * quat_log_v(quat_multi_v(quat_inverse_v(q[:, 1:]), quat_multi_v(q, Exp)[:, :-1]))),2)
  s2 = 0.5 * jnp.linalg.norm(A_corrected[:, 1:] - calculate_a(q)[1:])**2
  S1_loss.append(s1.astype(jnp.float64))
  S2_loss.append(s2.astype(jnp.int32))
  return s1 + s2

def orientation_from_W(W, Q, T):
  '''
  Calculating all orientation values sequentially. Also precomputing the value of exponential function.
  '''
  Exp = quat_exp_v(jnp.vstack((jnp.zeros(W.shape[1]-1), T*W[:,:-1]/2)))
  for i in range(W.shape[1]-1):
    Q = Q.at[:, i+1].set(quat_multi(Q[:, i], Exp[:, i]))
  return Exp, Q

# Functions to plot results
def plot_euler_angles(Q, vicd, trained_Q, prefix):
  '''
  Plot the euler angles to compare vicon data with predicted values and trained values.
  If vicon data is not available, then it is not plotted.
  Saves the plots as images names prefix_euler.jpg
  '''
  Q_to_V = jnp.zeros((3, Q.shape[1]))
  trained_Q_to_V = jnp.zeros((3, trained_Q[0, :].shape[0]))
  
  for v in range(Q.shape[1]):
    Q_to_V = Q_to_V.at[:, v].set(jnp.array(transforms3d.euler.quat2euler(Q[:,v])))
    trained_Q_to_V = trained_Q_to_V.at[:, v].set(jnp.array(transforms3d.euler.quat2euler(trained_Q[:,v])))

  if vicd:
    V = jnp.zeros((3, vicd['rots'].shape[2])) if vicd else None
    for v in range(vicd['rots'].shape[2]):
      V = V.at[:, v].set(jnp.array(transforms3d.euler.mat2euler(vicd['rots'][:,:,v]))) if vicd else None

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  fig.suptitle('Dataset {} Euler Angles'.format(prefix))
  titles = ['Euler angle X', 'Euler angle Y', 'Euler angle Z']
  for i in range(3):
    axes[i].set_title(titles[i])
    axes[i].plot(Q_to_V[i,:])
    axes[i].plot(trained_Q_to_V[i,:])
    if vicd: 
      axes[i].plot(V[i,:])
      axes[i].legend(['Predicted Values from W', "Trained Motion model", "Values from VICON"])
    else:
      axes[i].legend(['Predicted Values from W', "Trained Motion model", ])
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
    axes[i].plot(A[i+1,:])
    axes[i].plot(trained_A[i+1,:])
    axes[i].plot(A_corrected[i])
    axes[i].legend(["Predicted values from Q", "Trained Observation model", 'Values from IMU', ])
  plt.savefig('{}_A.jpg'.format(prefix))
  plt.close()

def plot_cost(S2_loss, dataset_number):
  fig, axes = plt.subplots(1, 1, figsize=(5, 5))
  axes.set_title('Loss for dataset {}'.format(dataset_number))
  axes.plot(S2_loss)
  plt.savefig('{}_cost.jpg'.format(dataset_number))
  plt.close()

def run_grad(Q, cost_fun, iterations=100, step_size=0.05):
  '''
  Running the gradient descent
  '''
  Q_iters = []
  Q_iters.append(Q[:,1:])
  start = datetime.now()
  for iter in tqdm(range(iterations)):
    C = jax.grad(cost_fun)(Q_iters[-1] + perturb)
    Q_iters.append(Q_iters[-1]-(step_size)*C)
  print(datetime.now()-start)
  return Q_iters

def create_vicd(timestamps, R):
  '''
  Create a data structure similar to the VICON data provided in the train dataset
  '''
  vicd = {'ts':timestamps, 'rots':np.zeros((3, 3, R.shape[1]))}
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

def create_panorama(vicd, camd):
  cam_time_map = {}
  HEIGHT, WIDTH = 960, 1920
  for t in camd['ts'][0]:
    temp = np.abs(vicd['ts'][0] - t)
    index = np.argmin(temp)
    cam_time_map[t] = index

  H, W, vert, hor = 240, 320, 45., 60.
  z_angles, x_angles = np.linspace(90-(vert/2), 90+(vert/2)-1, int(H)) * np.pi/180, np.linspace(90-(hor/2), 90+(hor/2)-1, int(W))*np.pi/180
  cartesian_temp, cartesian = np.ones((H, W, 4)), np.ones((H, W, 3))
  for i in range(z_angles.shape[0]):
    cartesian_temp[i, :, 0] = np.cos(x_angles)
  for i in range(x_angles.shape[0]):
    cartesian_temp[:, i, 1] = np.cos(z_angles)
  for i in range(z_angles.shape[0]):
    cartesian_temp[i, :, 2] = np.sin(x_angles)
  for i in range(x_angles.shape[0]):
    cartesian_temp[:, i, 3] = np.sin(z_angles)

  cartesian[:,:,0] = np.multiply(cartesian_temp[:,:,3], cartesian_temp[:,:,0])
  cartesian[:,:,1] = np.multiply(cartesian_temp[:,:,3], cartesian_temp[:,:,2])
  cartesian[:,:,2] = cartesian_temp[:,:,1]
  print('Created cartesian coordinates map')


  world_frame_cartesian = np.zeros((240, 320, 3, len(cam_time_map.keys())))
  for i in range(camd['cam'].shape[3]):
    world_frame_cartesian[:,:,:,i] = np.dot(cartesian, vicd['rots'][:, :, cam_time_map[camd['ts'][0][i]]])
  print('Created world frame')

  del cartesian_temp, cartesian, z_angles, x_angles, cam_time_map

  spherical_from_cartesian = np.zeros((H, W, 3, camd['cam'].shape[3]))

  spherical_from_cartesian_r = np.sqrt(np.sum(np.power(world_frame_cartesian, 2), axis=2))
  spherical_from_cartesian[:,:,0,:] = spherical_from_cartesian_r #rho => z
  spherical_from_cartesian[:,:,1,:] = np.arctan2(world_frame_cartesian[:,:,1,:], world_frame_cartesian[:,:,0,:]) #theta => x
  spherical_from_cartesian[:,:,2,:] = np.arccos(world_frame_cartesian[:,:,2,:]/spherical_from_cartesian_r) #phi => y
  del spherical_from_cartesian_r

  sx, sy = (2*np.pi/WIDTH), (np.pi/HEIGHT)
  spherical_from_cartesian[:, :, 1, :] += np.pi
  spherical_from_cartesian[:, :, 1, :] /= sx
  spherical_from_cartesian[:, :, 2, :] /= sy
  spherical_from_cartesian[:, :, 2, :] -= np.min(spherical_from_cartesian[:, :, 2, :])
  spherical_from_cartesian[:, :, 1, :] -= np.min(spherical_from_cartesian[:, :, 1, :])
  spherical_from_cartesian = spherical_from_cartesian.astype(np.int32)
  print('Created spherical projection map\nCreating image:')

  image = np.zeros((HEIGHT, WIDTH, 3)).astype(np.int32)
  for r in tqdm(range(camd['cam'].shape[3])):
    for i in range(camd['cam'].shape[0]):
      for j in range(camd['cam'].shape[1]):
        _, x, y = spherical_from_cartesian[i, j, :, r]
        image[y, x, :] = camd['cam'][i, j, :, r]
  print('Image created')
  return image

def create_panorama_lambert(vicd, camd):
  cam_time_map = {}
  for t in camd['ts'][0]:
    temp = np.abs(vicd['ts'][0] - t)
    index = np.argmin(temp)
    cam_time_map[t] = index

  H, W, vert, hor = 240, 320, 45., 60.
  z_angles, x_angles = np.linspace(90-(vert/2), 90+(vert/2)-1, int(H)) * np.pi/180, np.linspace(90-(hor/2), 90+(hor/2)-1, int(W))*np.pi/180
  cartesian_temp, cartesian = np.ones((H, W, 4)), np.ones((H, W, 3))
  for i in range(z_angles.shape[0]):
    cartesian_temp[i, :, 0] = np.cos(x_angles)
  for i in range(x_angles.shape[0]):
    cartesian_temp[:, i, 1] = np.cos(z_angles)
  for i in range(z_angles.shape[0]):
    cartesian_temp[i, :, 2] = np.sin(x_angles)
  for i in range(x_angles.shape[0]):
    cartesian_temp[:, i, 3] = np.sin(z_angles)

  cartesian[:,:,0] = np.multiply(cartesian_temp[:,:,3], cartesian_temp[:,:,0])
  cartesian[:,:,1] = np.multiply(cartesian_temp[:,:,3], cartesian_temp[:,:,2])
  cartesian[:,:,2] = cartesian_temp[:,:,1]
  print('Created cartesian coordinates map')


  world_frame_cartesian = np.zeros((240, 320, 3, len(cam_time_map.keys())))
  for i in range(camd['cam'].shape[3]):
    world_frame_cartesian[:,:,:,i] = np.dot(cartesian, vicd['rots'][:, :, cam_time_map[camd['ts'][0][i]]])
  print('Created world frame')

  lambert = np.zeros((240, 320, 2, len(cam_time_map.keys())))
  lambert[:,:,0,:] = np.sqrt(2/(1-world_frame_cartesian[:,:,2,:])) * world_frame_cartesian[:,:,0,:]
  lambert[:,:,1,:] = np.sqrt(2/(1-world_frame_cartesian[:,:,2,:])) * world_frame_cartesian[:,:,1,:]
  abs_lamb = 250*lambert
  abs_lamb = abs_lamb.astype(np.int32)
  abs_lamb = abs_lamb-np.min(abs_lamb)
  print('Created lambert azimuthal projection map\nCreating image:')

  image = np.zeros((1000, 1000, 3)).astype(np.int32)
  for r in tqdm(range(camd['cam'].shape[3])):
    for i in range(240):
      for j in range(320):
        x, y = abs_lamb[i, j, :, r]
        image[y, x, :] = camd['cam'][i, j, :, r]
  plt.imshow(image)
  plt.close()
  print('Image created')
  return image

dataset_number = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
dataset_folder = ['trainset', 'trainset', 'trainset', 'trainset', 'trainset', 'trainset', 'trainset', 'trainset', 'trainset', 'testset', 'testset', ]
has_cam = [True, True, False,  False,  False,  False,  False, True, True, True, True]
has_vic = [True, True, True, True, True, True, True, True, True, False,  False]

if __name__ == '__main__':
# for i in range(len(dataset_number)):
  while True:
    i = int(input('Enter dataset to run for (1-11):')) - 1
    if i<0 or i>11:
      print('Invalid dataset entry. Enter value between 1 and 11.')
      continue
    print('Calculations for dataset {}'.format(dataset_number[i]))
    start = datetime.now()
    S1_loss, S2_loss = [], []
    camd, imud, vicd = load_data(dataset_number[i], dataset_folder[i], has_cam[i], has_vic[i])

    print('Calculating constants and correcting bias')
    scale_factor_gyro = scale_factor(Vref_gyro, 10, sensitivity_gyro)
    scale_factor_acc = scale_factor(Vref_acc, 10, sensitivity_acc)

    imu_biases = calculate_bias(threshold_index+1, imud['vals'])
    imud_new = imud.copy()
    # Correcting data using bias and scale factor
    A_corrected = IMU_value(imud['vals'][0:3, :], imu_biases[:3], sensitivity_acc, Vref_acc, 10)
    W_corrected = IMU_value(imud['vals'][3:, :], imu_biases[3:], sensitivity_gyro, Vref_gyro, 10)
    W_corrected = W_corrected.at[jnp.where(W_corrected==0)].set(perturb)
    # Multiplying and adding gravity value to show gravity units
    A_corrected = jnp.multiply(A_corrected * g, jnp.array([-1, -1, 1])[:, jnp.newaxis]) + jnp.array([0, 0, g])[:, jnp.newaxis]
    # Assigning corrected values to separate variables
    Ax, Ay, Az = A_corrected

    print('Calculating motion model from IMU data')
    # Grouping Ws
    W = jnp.vstack((W_corrected[1], W_corrected[2], W_corrected[0]))
    Q = jnp.zeros((4, W.shape[1]), dtype=np.float64)
    Q = Q.at[0, 0].set(1)
    # Calculation of Exponential and Q 
    Exp, Q = orientation_from_W(jnp.vstack((W_corrected[1], W_corrected[2], W_corrected[0])), Q, imud_new['ts'][0][1:] - imud_new['ts'][0][:-1])
    A = calculate_a(Q)

    # Gradient Descent
    print('Started training')
    Q_iters = jnp.array(run_grad(Q, cost_fun, iterations, step_size))
    print('Loss started at: {}'.format(S2_loss[0].astype(np.int32)))
    print('Loss ended at: {}'.format(S2_loss[-1].astype(np.int32)))
    
    # Plotting the graph
    print('Plotting and saving graphs')
    R = plot_euler_angles(Q, vicd, Q_iters[-1], dataset_number[i])
    plot_angular_acceleration(A, A_corrected, calculate_a(Q_iters[-1]), dataset_number[i])
    vicd = create_vicd(imud['ts'][:, :R.shape[1]], R) if not has_vic[i] else vicd
    print('Dataset {} processed in {}'.format(dataset_number[i], str(datetime.now()-start)))
    
    if has_cam[i]:
      print('Starting construction of panorama')
      image = create_panorama_lambert(vicd, camd)
      Image.fromarray(image.astype(np.uint8)).save('lambert_panorama_{}.jpg'.format(dataset_number[i]))
      print('Lambert image stored')
      image = create_panorama(vicd, camd)
      Image.fromarray(image.astype(np.uint8)).save('panorama_{}.jpg'.format(dataset_number[i]))
      print('Image stored')
      
    print('Time elapsed: ', datetime.now()-start)
    break