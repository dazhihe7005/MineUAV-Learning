"""Real MuJoCo generalized force and native C++ guard, no flight."""
from pathlib import Path
import subprocess
import numpy as np
import pytest
import mujoco
from .test_interfaces import ROOT,fixture,module


def setup():
    m,d=fixture();a=module("actuator_adapter").ActuatorAdapter(m,ROOT/"models/rotor_config.json")
    return m,d,a


@pytest.mark.parametrize("controls",[[0,0,0,0],[1,1,1,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[.12,.35,.76,.94]])
def test_physical_wrench_matches_actual_mujoco_generalized_force(controls):
    m,d,a=setup();b=module("allocation_math").effectiveness(a)
    d.ctrl[:]=a.apply(controls,True);mujoco.mj_forward(m,d)
    force=d.qfrc_actuator[:3]
    moment=d.qfrc_actuator[3:]-np.cross(m.body("mine_uav").ipos,force)
    actual=np.r_[moment*np.array([1,-1,-1]),force*np.array([1,-1,-1])]
    np.testing.assert_allclose(b@controls*(a.kf*a.u_max),actual,atol=1e-10)


def test_original_filter_loses_only_yaw_and_patch_keeps_true_matrix():
    _,_,a=setup();math=module("allocation_math");b=math.effectiveness(a)
    old=math.filtered(b,False);new=math.filtered(b,True)
    np.testing.assert_equal(old[2],0)
    assert np.linalg.matrix_rank(old)==3 and np.linalg.matrix_rank(new)==4
    np.testing.assert_allclose(new,b,atol=0,rtol=0)
    np.testing.assert_equal(old[[0,1,3,4,5]],new[[0,1,3,4,5]])


@pytest.mark.parametrize("yaw",[-.2,0.,.2])
def test_hover_and_bidirectional_yaw_preserve_force_roll_pitch(yaw):
    _,_,a=setup();b=module("allocation_math").effectiveness(a)*(a.kf*a.u_max)
    w=np.array([-68.67,0,0,yaw]);c=np.linalg.solve(b[[5,0,1,2]],w)
    assert np.all((c>0)&(c<1))
    np.testing.assert_allclose(b[[5,0,1,2]]@(a.apply(c,True)/a.u_max),w,atol=1e-12)
    if yaw:
        hover=np.linalg.solve(b[[5,0,1,2]],[-68.67,0,0,0])
        np.testing.assert_equal(np.sign(c-hover),np.sign(yaw)*np.array([-1,1,-1,1]))


def test_nominal_roll_pitch_collective_pseudoinverse_is_unchanged():
    _,_,a=setup();math=module("allocation_math");b=math.effectiveness(a)
    np.testing.assert_allclose(np.linalg.pinv(math.filtered(b,False))[:,[0,1,5]],np.linalg.pinv(math.filtered(b,True))[:,[0,1,5]],atol=1e-12)


def test_unreachable_yaw_is_not_claimed_achieved_after_motor_limits():
    _,_,a=setup();b=module("allocation_math").effectiveness(a)*(a.kf*a.u_max)
    c=np.linalg.solve(b[[5,0,1,2]],[-68.67,0,0,100])
    assert np.any((c<0)|(c>1))
    with pytest.raises(ValueError):a.apply(c,True)
    feasible=np.clip(c,0,1)
    assert abs((b@feasible)[2])<100


def test_uniform_scale_inverse_requires_different_domain_not_full_px4_box():
    _,_,a=setup();b=module("allocation_math").effectiveness(a)
    c=np.array([.1,.2,.7,1.]);v=c/4
    np.testing.assert_allclose((4*b)@v,b@c,atol=1e-12)
    np.testing.assert_allclose(a.apply(4*v,True),a.apply(c,True))
    with pytest.raises(ValueError):a.apply(4*np.full(4,.5),True)


def test_native_guard_retains_independent_axis_and_rejects_uncontrollable_inputs(tmp_path):
    header=Path(__file__).parent/"compat/MineUavSitlYawAuthority.hpp"
    assert header.is_file(),"native independent-yaw guard missing"
    code=r'''
    #include "MineUavSitlYawAuthority.hpp"
    #include <cassert>
    #include <limits>
    struct Matrix {float v[6][4]{};float operator()(int r,int c)const{return v[r][c];}};
    int main(){
      Matrix m;float r[4]={-1,1,1,-1},p[4]={-1,-1,1,1},y[4]={-1,1,-1,1};
      for(int i=0;i<4;i++){m.v[0][i]=.22f*r[i];m.v[1][i]=.22f*p[i];m.v[2][i]=.017207708f*y[i];m.v[5][i]=-1;}
      assert(mineuavSitlYawAuthority(m,4));
      assert(!mineuavSitlYawAuthority(m,3));
      Matrix bad=m;for(int i=0;i<4;i++)bad.v[2][i]=0;assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[2][i]=.017f;assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[2][i]=.017f*r[i];assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[2][i]=.017f*r[i]+.00001f*y[i];assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[2][i]=1e-9f*y[i];assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;bad.v[2][0]=std::numeric_limits<float>::quiet_NaN();assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;bad.v[3][0]=.1f;assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[1][i]=bad.v[0][i];assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<6;i++)bad.v[i][0]=0;assert(!mineuavSitlYawAuthority(bad,4));
      bad=m;for(int i=0;i<4;i++)bad.v[0][i]*=.1f;assert(!mineuavSitlYawAuthority(bad,4));
      MineUavSitlMotorHealth health;
      assert(!health.allowed());
      assert(health.observe(false,0,0) && health.allowed());
      assert(!health.observe(false,0,0));
      // Raw detector fault revokes authority even if upstream failure mode is Ignore.
      assert(health.observe(true,0,0) && !health.allowed());
      assert(health.observe(false,0,0) && health.allowed());
      assert(health.observe(false,1,0) && !health.allowed());
      assert(health.observe(false,0,0) && health.allowed());
      assert(health.observe(false,0,2) && !health.allowed());
      assert(!health.observe(false,0,2));
    }'''
    source=tmp_path/"guard.cpp";source.write_text(code)
    subprocess.run(["c++","-std=c++14","-Wall","-Wextra","-Werror","-Wdouble-promotion","-I",str(header.parent),str(source),"-o",str(tmp_path/"guard")],check=True)
    subprocess.run([str(tmp_path/"guard")],check=True)
