"""Pinned rotor effectiveness and conservative SITL weak-yaw exception audit."""
import numpy as np


def effectiveness(adapter):
    axis=np.array([0.,0.,-1.]);b=np.zeros((6,4))
    params=adapter.px4_geometry()
    for i,r in enumerate(adapter.arms_frd):
        ct=params[f"CA_ROTOR{i}_CT"];km=params[f"CA_ROTOR{i}_KM"]
        b[:3,i]=ct*np.cross(r,axis)-ct*km*axis;b[3:,i]=ct*axis
    return b


def independent_yaw(b):
    b=np.asarray(b,dtype=float)
    if b.shape!=(6,4) or not np.isfinite(b).all():return False
    if np.max(np.abs(b[3:5]))>1e-6 or np.max(np.abs(b[5]+1))>1e-6:return False
    if not (-.05<=b[2].min()<=-.01 and .01<=b[2].max()<=.05):return False
    if np.any(np.max(np.abs(b[:2]),axis=1)<=.05):return False
    rows=b[[5,0,1,2]];norm=np.linalg.norm(rows,axis=1)
    if np.any(norm**2<1e-6):return False
    bases=[]
    for vector in rows/norm[:,None]:
        v=vector.copy()
        for _ in range(2):
            for earlier in bases:v-=np.dot(v,earlier)*earlier
        residual=np.linalg.norm(v)
        if residual<.5:return False
        bases.append(v/residual)
    trim=np.linalg.solve(rows,[-1,0,0,0])
    return bool(np.all((trim>.1)&(trim<.4)))


def filtered(b,repaired):
    b=np.asarray(b,dtype=float)
    if b.shape!=(6,4) or not np.isfinite(b).all():raise ValueError("finite6x4 rotor matrix required")
    keep=repaired and independent_yaw(b)
    result=b.copy()
    for row in range(6):
        if np.max(np.abs(b[row]))<=.05 and not (row==2 and keep):result[row]=0
    return result


def physical_audit(adapter):
    b=effectiveness(adapter);after=filtered(b,True);physical=b*(adapter.kf*adapter.u_max)
    rows=physical[[5,0,1,2]];hover=np.linalg.solve(rows,[-68.67,0,0,0])
    normalized=b[[5,0,1,2]]/np.linalg.norm(b[[5,0,1,2]],axis=1)[:,None]
    return {"raw_px4_matrix":b.tolist(),"old_filtered_matrix":filtered(b,False).tolist(),
        "new_filtered_matrix":after.tolist(),"physical_com_frd_matrix_per_normalized_motor":physical.tolist(),
        "axis_order":["Tx","Ty","Tz","Fx","Fy","Fz"],"physical_units":"Nm/N; FRD at COM",
        "original_rank":int(np.linalg.matrix_rank(filtered(b,False))),"new_rank":int(np.linalg.matrix_rank(after)),
        "row_normalized_condition_number":float(np.linalg.cond(normalized)),
        "physical_condition_number_dimensionful_not_coordinate_invariant":float(np.linalg.cond(rows)),
        "hover_controls":hover.tolist(),"hover_wrench_fz_tx_ty_tz":(rows@hover).tolist(),
        "maximum_motor_thrust_n":float(adapter.kf*adapter.u_max),"maximum_motor_reaction_torque_nm":float(adapter.km*adapter.u_max),
        "zero_yaw_nominal_mix_delta_max":float(np.max(np.abs(np.linalg.pinv(filtered(b,False))[:,[0,1,5]]-np.linalg.pinv(after)[:,[0,1,5]]))),
        "physical_mapping_equivalence":"Exact B entries, adapter c→omega²=u_max*c and box[0,1] unchanged; linear map equality on basis implies entire feasible motor box equality, not only hover.",
        "scheme_a":"Uniform B'=sB requires c=s*v and v∈[0,1/s], not stock[0,1]. PX4 also normalizes mixer axes internally; no full allocator/range/saturation equivalence proved. Rejected for flight.",
        "scheme_b":"SITL-only guarded retention of independent weak yaw row, with zero-moment interior trim; original PID/EKF/normalization/desaturation/min/max unchanged."}
