"""R8 uses the released R7 boundary and original training initialization."""
from .short_goal_env import make_short_scheduler
from .task_accounting import DistanceTaskAccountingAdapter
from .distance_training import initialize_distance_training


def make_repaired_scheduler(prereg,upstream,*,seed):
    return DistanceTaskAccountingAdapter(make_short_scheduler(prereg,upstream,seed=seed),prereg)


def initialize_repaired_training(model,scheduler,manager,*,identity,seed):
    initialization,event,probe=initialize_distance_training(model,scheduler,manager,identity=identity,seed=seed)
    initialization['execution_profile']='DISTANCE_CURRICULUM_R8_REPAIRED_V1'
    return initialization,event,probe
