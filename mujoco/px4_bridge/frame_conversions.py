"""Proper rotations: MuJoCo ENU/FLU to PX4 NED/FRD, quaternions wxyz."""
import mujoco
import numpy as np

ENU_TO_NED = np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]])
FLU_TO_FRD = np.diag([1.,-1.,-1.])

def vector(value):
    v=np.asarray(value,dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError("expected finite vector3")
    return v

def enu_to_ned(v):
    return ENU_TO_NED @ vector(v)

def flu_to_frd(v):
    return FLU_TO_FRD @ vector(v)

def rotation(q):
    q=np.asarray(q,dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-6:
        raise ValueError("expected unit finite wxyz quaternion")
    r=np.empty(9);mujoco.mju_quat2Mat(r,q)
    return r.reshape(3,3)

def matrix_quaternion(r):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.ascontiguousarray(r).ravel())
    return q if q[0]>=0 else -q

def attitude_to_px4(q):
    return matrix_quaternion(ENU_TO_NED @ rotation(q) @ FLU_TO_FRD)

def attitude_from_px4(q):
    return matrix_quaternion(ENU_TO_NED.T @ rotation(q) @ FLU_TO_FRD.T)
