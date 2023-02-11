from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)

'''Create panoramas'''


def create_panorama(vicd, camd):
    cam_time_map = {}
    HEIGHT, WIDTH = 960, 1920
    for t in camd['ts'][0]:
        temp = np.abs(vicd['ts'][0] - t)
        index = np.argmin(temp)
        cam_time_map[t] = index

    H, W, vert, hor = 240, 320, 45., 60.
    z_angles, x_angles = np.linspace(90-(vert/2), 90+(vert/2)-1, int(
        H)) * np.pi/180, np.linspace(90-(hor/2), 90+(hor/2)-1, int(W))*np.pi/180
    cartesian_temp, cartesian = np.ones((H, W, 4)), np.ones((H, W, 3))
    for i in range(z_angles.shape[0]):
        cartesian_temp[i, :, 0] = np.cos(x_angles)
    for i in range(x_angles.shape[0]):
        cartesian_temp[:, i, 1] = np.cos(z_angles)
    for i in range(z_angles.shape[0]):
        cartesian_temp[i, :, 2] = np.sin(x_angles)
    for i in range(x_angles.shape[0]):
        cartesian_temp[:, i, 3] = np.sin(z_angles)

    cartesian[:, :, 0] = np.multiply(
        cartesian_temp[:, :, 3], cartesian_temp[:, :, 0])
    cartesian[:, :, 1] = np.multiply(
        cartesian_temp[:, :, 3], cartesian_temp[:, :, 2])
    cartesian[:, :, 2] = cartesian_temp[:, :, 1]
    print('Created cartesian coordinates map')

    world_frame_cartesian = np.zeros((240, 320, 3, len(cam_time_map.keys())))
    for i in range(camd['cam'].shape[3]):
        world_frame_cartesian[:, :, :, i] = np.dot(
            cartesian, vicd['rots'][:, :, cam_time_map[camd['ts'][0][i]]])
    print('Created world frame')

    del cartesian_temp, cartesian, z_angles, x_angles, cam_time_map

    spherical_from_cartesian = np.zeros((H, W, 3, camd['cam'].shape[3]))

    spherical_from_cartesian_r = np.sqrt(
        np.sum(np.power(world_frame_cartesian, 2), axis=2))
    spherical_from_cartesian[:, :, 0,
                             :] = spherical_from_cartesian_r  # rho => z
    spherical_from_cartesian[:, :, 1, :] = np.arctan2(
        world_frame_cartesian[:, :, 1, :], world_frame_cartesian[:, :, 0, :])  # theta => x
    spherical_from_cartesian[:, :, 2, :] = np.arccos(
        world_frame_cartesian[:, :, 2, :]/spherical_from_cartesian_r)  # phi => y
    del spherical_from_cartesian_r

    sx, sy = (2*np.pi/WIDTH), (np.pi/HEIGHT)
    spherical_from_cartesian[:, :, 1, :] += np.pi
    spherical_from_cartesian[:, :, 1, :] /= sx
    spherical_from_cartesian[:, :, 2, :] /= sy
    spherical_from_cartesian[:, :, 2,
                             :] -= np.min(spherical_from_cartesian[:, :, 2, :])
    spherical_from_cartesian[:, :, 1,
                             :] -= np.min(spherical_from_cartesian[:, :, 1, :])
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
    z_angles, x_angles = np.linspace(90-(vert/2), 90+(vert/2)-1, int(
        H)) * np.pi/180, np.linspace(90-(hor/2), 90+(hor/2)-1, int(W))*np.pi/180
    cartesian_temp, cartesian = np.ones((H, W, 4)), np.ones((H, W, 3))
    for i in range(z_angles.shape[0]):
        cartesian_temp[i, :, 0] = np.cos(x_angles)
    for i in range(x_angles.shape[0]):
        cartesian_temp[:, i, 1] = np.cos(z_angles)
    for i in range(z_angles.shape[0]):
        cartesian_temp[i, :, 2] = np.sin(x_angles)
    for i in range(x_angles.shape[0]):
        cartesian_temp[:, i, 3] = np.sin(z_angles)

    cartesian[:, :, 0] = np.multiply(
        cartesian_temp[:, :, 3], cartesian_temp[:, :, 0])
    cartesian[:, :, 1] = np.multiply(
        cartesian_temp[:, :, 3], cartesian_temp[:, :, 2])
    cartesian[:, :, 2] = cartesian_temp[:, :, 1]
    print('Created cartesian coordinates map')

    world_frame_cartesian = np.zeros((240, 320, 3, len(cam_time_map.keys())))
    for i in range(camd['cam'].shape[3]):
        world_frame_cartesian[:, :, :, i] = np.dot(
            cartesian, vicd['rots'][:, :, cam_time_map[camd['ts'][0][i]]])
    print('Created world frame')

    lambert = np.zeros((240, 320, 2, len(cam_time_map.keys())))
    lambert[:, :, 0, :] = np.sqrt(
        2/(1-world_frame_cartesian[:, :, 2, :])) * world_frame_cartesian[:, :, 0, :]
    lambert[:, :, 1, :] = np.sqrt(
        2/(1-world_frame_cartesian[:, :, 2, :])) * world_frame_cartesian[:, :, 1, :]
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
