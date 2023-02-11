from constants import *
import jax
import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)


'''Operations for quaternions'''


@jax.jit
def quat_multi(q, p):
    '''
    Multiply 2 quaternions
    '''
    qs, ps, qv, pv = q[0], p[0], q[1:], p[1:]
    return jnp.hstack(((qs*ps) - jnp.dot(qv.T, pv), (qs*pv) + (ps*qv) + jnp.cross(qv, pv)))


@jax.jit
def quat_exp_v(q):
    '''
    Vectorized exponential of a quaternion
    '''
    qs, qv = q[0, :], q[1:, :]
    return jnp.exp(qs)*jnp.vstack((jnp.cos(jnp.linalg.norm(qv, axis=0)), (qv/jnp.linalg.norm(qv, axis=0))*jnp.sin(jnp.linalg.norm(qv, axis=0))))

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
    qs, ps, qv, pv = q[0, :], p[0, :], q[1:, :], p[1:, :]
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
