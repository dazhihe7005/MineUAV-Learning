"""Conservative Phase3C-R predicates; missing evidence never grants PASS."""
import numpy as np

EPOCH_USEC=1_000_000
END_USEC=21_000_000
EXPECTED_COUNTS={"HIL_SENSOR":5000,"HIL_GPS":200,"ODOMETRY":1000}
DIRECT_MIN_SAMPLES=900
DIRECT_MAX_GAP_USEC=250_000
FIELD_TOLERANCE=2e-6
CHECK_NAMES=("A_producer","B_send","C_direct_receipt","D_ev_position","E_ev_velocity",
             "F_gnss","G_duration","H_final_identity","I_safety","J_disarmed_zero",
             "K_log_integrity","L_cleanup_history")


def aggregate_checks(checks):
    if not set(CHECK_NAMES)<=set(checks):return "BLOCKED"
    if all(value=="PASS" for value in checks.values()):return "PASS"
    return "FAIL" if "FAIL" in checks.values() else "BLOCKED"


def assess_boundary(raw):
    seconds=raw.get("simulation_seconds",np.nan)
    physical=raw.get("final_physical_state",{}).get("time_s",np.nan)
    passed=bool(raw.get("complete") is True and np.isfinite(seconds) and np.isfinite(physical)
                and 0<=seconds<=20. and 0<=physical<=20.
                and abs(seconds-20.)<1e-9 and abs(physical-seconds)<1e-9
                and all(raw.get("messages_sent",{}).get(k)==v for k,v in EXPECTED_COUNTS.items()))
    return {"status":"PASS" if passed else "FAIL","simulation_seconds":seconds,
            "physical_seconds":physical,"expected_counts":EXPECTED_COUNTS,
            "actual_counts":raw.get("messages_sent",{}),
            "comparison_guard_s":1e-6,"sensor_cycles":5000,
            "reason":None if passed else "incomplete, overshoot, missing cycle or count mismatch"}


def vector(data,prefix,size=3):
    return np.column_stack([data[f"{prefix}[{i}]"] for i in range(size)])


def assess_direct(data,*,end_usec):
    required=["timestamp","timestamp_sample","pose_frame","velocity_frame","quality","reset_counter"]
    for prefix,size in (("position",3),("velocity",3),("angular_velocity",3),("q",4),
                        ("position_variance",3),("orientation_variance",3),("velocity_variance",3)):
        required.extend(f"{prefix}[{i}]" for i in range(size))
    if not data or any(k not in data for k in required):
        return {"status":"BLOCKED","samples":0,"reason":"missing vehicle_visual_odometry or required direct fields"}
    arrays={k:np.asarray(data[k],dtype=float) for k in required};n=len(arrays["timestamp"])
    structural=all(v.ndim==1 and len(v)==n and np.isfinite(v).all() for v in arrays.values())
    passed=False
    if structural and n>=DIRECT_MIN_SAMPLES:
        stamp=arrays["timestamp"];sample=arrays["timestamp_sample"]
        passed=bool(np.all(np.diff(stamp)>0) and np.all(np.diff(sample)>0)
            and np.all((sample>=EPOCH_USEC)&(sample<=stamp))
            and stamp[0]-EPOCH_USEC<=1_000_000
            and np.max(np.diff(sample))<=DIRECT_MAX_GAP_USEC
            and np.max(np.diff(stamp))<=DIRECT_MAX_GAP_USEC
            and 0<=end_usec-sample[-1]<=DIRECT_MAX_GAP_USEC
            and 0<=end_usec-stamp[-1]<=DIRECT_MAX_GAP_USEC
            and np.all(arrays["pose_frame"]==1) and np.all(arrays["velocity_frame"]==3)
            and np.all(arrays["quality"]==100) and np.all(arrays["reset_counter"]==0)
            and np.all(np.abs(np.linalg.norm(vector(arrays,"q",4),axis=1)-1.)<=1e-5)
            and all(np.all(np.abs(vector(arrays,prefix)-.01)<=FIELD_TOLERANCE)
                    for prefix in ("position_variance","orientation_variance","velocity_variance")))
    return {"status":"PASS" if passed else "FAIL","samples":n,
            "reason":None if passed else "invalid, truncated, stale, gap, quaternion, frame or covariance evidence",
            "minimum_samples":DIRECT_MIN_SAMPLES,"maximum_gap_usec":DIRECT_MAX_GAP_USEC,
            "first_publication_usec":int(arrays["timestamp"][0]) if structural and n else None,
            "last_publication_usec":int(arrays["timestamp"][-1]) if structural and n else None,
            "last_sample_usec":int(arrays["timestamp_sample"][-1]) if structural and n else None,
            "max_publication_gap_usec":int(np.max(np.diff(arrays["timestamp"]))) if structural and n>1 else None,
            "pose_frame":"NED (1)","velocity_frame":"BODY_FRD (3)",
            "covariance":"receiver retains three diagonal variances per block, not all MAVLink cross-covariances"}


def compare_payloads(sent,data):
    """Exact logical-clock correspondence, never future or nearest interpolated data.

    Pinned SimulatorMavlink sets hrt from HIL_SENSOR before receiving ODOMETRY.
    We test that relation by exact matching; host monotonic has no calibrated
    offset to PX4's lockstep clock. This is not a physical transport-delay test.
    """
    base={"status":"BLOCKED","paired_samples":0,"host_transport_latency":"NOT TESTED",
          "clock_semantics":"exact source/HIL_SENSOR logical time matched to receiver timestamp_sample; no host-to-PX4 delay claim"}
    if not sent or not data:return base,None
    source=np.asarray([r["sample_time_usec"] for r in sent],dtype=np.int64)
    received=np.asarray(data["timestamp_sample"],dtype=np.int64)
    if np.any(np.diff(source)<=0):return dict(base,status="FAIL",reason="nonmonotonic source"),None
    indices=np.searchsorted(source,received)
    valid=indices<len(source)
    valid[valid]&=source[indices[valid]]==received[valid]
    if not valid.all():return dict(base,status="BLOCKED",reason="exact logical clock correspondence unverified"),None
    fields=[sent[int(i)].get("message_fields") for i in indices]
    if not all(fields):return dict(base,reason="missing successful-send payload witness"),None
    errors={};matched={"time_s":(received-EPOCH_USEC)/1e6}
    for prefix,keys in (("position",("x","y","z")),("velocity",("vx","vy","vz")),
                        ("angular_velocity",("rollspeed","pitchspeed","yawspeed"))):
        a=np.asarray([[row[key] for key in keys] for row in fields]);b=vector(data,prefix)
        errors[prefix]=float(np.max(np.abs(a-b)))
        matched["source_"+prefix]=a;matched["received_"+prefix]=b
    a=np.asarray([row["q"] for row in fields]);b=vector(data,"q",4)
    errors["quaternion"]=float(np.max(np.minimum(np.max(np.abs(a-b),axis=1),np.max(np.abs(a+b),axis=1))))
    for prefix,key,diag in (("position_variance","pose_covariance",(0,6,11)),
                            ("orientation_variance","pose_covariance",(15,18,20)),
                            ("velocity_variance","velocity_covariance",(0,6,11))):
        a=np.asarray([[row[key][i] for i in diag] for row in fields]);b=vector(data,prefix)
        errors[prefix]=float(np.max(np.abs(a-b)))
    frame_ok=all(row["frame_id"]==1 and row["child_frame_id"]==12 for row in fields)
    extension_ok=all(row["reset_counter"]==int(data["reset_counter"][i]) and row["quality"]==int(data["quality"][i]) for i,row in enumerate(fields))
    passed=frame_ok and extension_ok and all(np.isfinite(e) and e<=FIELD_TOLERANCE for e in errors.values())
    return dict(base,status="PASS" if passed else "FAIL",paired_samples=len(indices),
                max_absolute_errors=errors,field_tolerance=FIELD_TOLERANCE,
                publication_minus_receipt_usec={"median":float(np.median(np.asarray(data["timestamp"])-received)),
                                                "max":int(np.max(np.asarray(data["timestamp"])-received))}),matched
