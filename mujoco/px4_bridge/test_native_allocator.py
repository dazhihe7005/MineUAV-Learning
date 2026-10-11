"""Compile pinned PX4's selected SequentialDesaturation without starting SITL."""
from pathlib import Path
import subprocess
import numpy as np
from .test_interfaces import module,fixture,ROOT


def test_actual_px4_normalization_preserves_hold_and_has_bounded_yaw(tmp_path):
    px4=Path.home()/"PX4-Autopilot-mineuav-yaw";build=px4/"build/px4_sitl_mineuav_yaw"
    m,_=fixture();a=module("actuator_adapter").ActuatorAdapter(m,ROOT/"models/rotor_config.json")
    b=module("allocation_math").effectiveness(a)
    assignments="\n".join(f"b({r},{c})={b[r,c]:.12f}f;" for r in range(6) for c in range(4))
    source=r'''
    #include "ControlAllocationSequentialDesaturation.hpp"
    #include "MineUavSitlYawAuthority.hpp"
    #include <cassert>
    #include <cmath>
    #include <initializer_list>
    #include <map>
    #include <string>
    // Only command-console registry symbols are no-ops; allocation, ModuleParams,
    // normalization, desaturation and the real default parameter store are linked.
    using AppMap=std::map<std::string,int(*)(int,char**)>;
    void init_app_map(AppMap&) {}
    void list_builtins(AppMap&) {}
    extern void param_init();
    int main(){
      param_init();
      param_control_autosave(false);
      int32_t air=-1,method=-1;
      assert(param_get(param_find("MC_AIRMODE"),&air)==0 && air==0);
      assert(param_get(param_find("CA_METHOD"),&method)==0 && method==2);
      ActuatorEffectiveness::EffectivenessMatrix b{};
      ASSIGNMENTS
      assert(mineuavSitlYawAuthority(b,4));
      auto old=b;old.row(2)=0.f;
      ControlAllocationSequentialDesaturation fixed,legacy;
      ControlAllocation::ActuatorVector zero{};
      fixed.setNormalizeRPY(true);legacy.setNormalizeRPY(true);
      fixed.setEffectivenessMatrix(b,zero,zero,4,true);
      legacy.setEffectivenessMatrix(old,zero,zero,4,true);
      matrix::Vector<float,6> request{};request(5)=-.44f;
      // Hold + small roll/pitch retain identical outputs at zero yaw.
      for(float rp:{0.f,.03f,-.03f}){
        request(0)=rp;request(1)=-rp;request(2)=0;
        fixed.setControlSetpoint(request);legacy.setControlSetpoint(request);
        fixed.allocate();legacy.allocate();
        for(int i=0;i<4;i++)assert(std::fabs(fixed.getActuatorSetpoint()(i)-legacy.getActuatorSetpoint()(i))<1e-5f);
      }
      request(0)=request(1)=0;
      for(float yaw:{-.05f,.05f}){
        request(2)=yaw;fixed.setControlSetpoint(request);fixed.allocate();fixed.clipActuatorSetpoint();
        auto c=fixed.getActuatorSetpoint();matrix::Vector<float,6> w=b*c;
        assert(w(2)*yaw>0.f);
        assert(std::fabs(w(0))<1e-5f && std::fabs(w(1))<1e-5f);
        assert(std::fabs(fixed.getAllocatedControl()(5)+.44f)<1e-5f);
        for(int i=0;i<4;i++)assert(c(i)>0 && c(i)<1);
      }
      // Unreachable/extreme normalized requests stay inside motor limits.
      for(float yaw:{-100.f,100.f}){
        request(0)=100.f;request(2)=yaw;
        fixed.setControlSetpoint(request);fixed.allocate();fixed.clipActuatorSetpoint();
        for(int i=0;i<4;i++)assert(fixed.getActuatorSetpoint()(i)>=0 && fixed.getActuatorSetpoint()(i)<=1);
        assert(std::fabs(fixed.getAllocatedControl()(2))<100.f);
      }
    }'''.replace("ASSIGNMENTS",assignments)
    cpp=tmp_path/"allocator.cpp";cpp.write_text(source);lib=px4/"src/lib/control_allocation/control_allocation"
    includes=[px4/"src/lib",px4/"src/lib/matrix",px4/"platforms/common",px4/"src/modules",build,build/"src/lib",px4/"platforms/common/include",px4/"platforms/posix/include",px4/"src/include",lib,
              Path(__file__).parent/"compat"]
    compiled=subprocess.run(["c++","-std=c++17","-D__PX4_POSIX","-D__EXPORT=__attribute__((visibility(\"default\")))",*(f"-I{p}" for p in includes),str(cpp),
        str(lib/"ControlAllocation.cpp"),str(lib/"ControlAllocationPseudoInverse.cpp"),str(lib/"ControlAllocationSequentialDesaturation.cpp"),
        "-Wl,--start-group",*(str(p) for p in sorted(build.rglob("*.a"))),"-Wl,--end-group","-lpthread","-ldl","-o",str(tmp_path/"allocator")],capture_output=True,text=True)
    assert compiled.returncode==0,compiled.stderr
    run=subprocess.run([str(tmp_path/"allocator")],capture_output=True,text=True,timeout=10)
    assert run.returncode==0,run.stdout+run.stderr
