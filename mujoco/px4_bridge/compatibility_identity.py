"""Explicit allowlist for the isolated, reviewed SITL binary. No bypass flag."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from .diagnostics import sha256

BASE="8f9b91b2bdd3a65952601a084c3aa35e96fee59d"
COMPAT=Path(__file__).parent/"compat"
MANIFEST=COMPAT/"px4_yaw_build_identity.json"
BUILD="build/px4_sitl_mineuav_yaw"
ALLOWED={"boards/px4/sitl/mineuav_yaw.px4board","src/modules/control_allocator/ControlAllocator.cpp",
         "src/modules/control_allocator/CMakeLists.txt","src/modules/control_allocator/MineUavSitlYawAuthority.hpp"}
ALLOWED.add("src/modules/control_allocator/ControlAllocator.hpp")


def git(root,*args):return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()


def tree_hash(root):
    """Content hash of startup files, including referenced symlink contents."""
    root=Path(root);h=hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode()+b"\0")
        h.update(sha256(path).encode()+b"\n")
    return h.hexdigest()


def observed_identity(root):
    """Collect only; never authorizes execution without the allowlist."""
    root=Path(root).resolve();build=root/BUILD;binary=build/"bin/px4"
    patch=subprocess.check_output(["git","-C",str(root),"diff",BASE,"HEAD","--",*sorted(ALLOWED)])
    return {"schema":1,"base_commit":BASE,"commit":git(root,"rev-parse","HEAD"),
        "build":BUILD,"binary_sha256":sha256(binary),
        "build_version_header_sha256":sha256(build/"src/lib/version/build_git_version.h"),
        "patch_sha256":hashlib.sha256(patch).hexdigest(),"rootfs_sha256":tree_hash(build/"etc"),
        "boardconfig_sha256":sha256(build/"boardconfig"),
        "simulator_source_sha256":sha256(root/"src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp")}


def verified_identity(root,manifest_path):
    try:
        approved=json.loads(MANIFEST.read_text());supplied=json.loads(Path(manifest_path).read_text())
        root=Path(root).resolve();build=root/BUILD
        if supplied!=approved or approved["base_commit"]!=BASE:raise ValueError("not the approved manifest")
        if git(root,"diff","--name-only","HEAD"):raise ValueError("dirty PX4 source")
        subprocess.run(["git","-C",str(root),"merge-base","--is-ancestor",BASE,"HEAD"],check=True,capture_output=True)
        if set(git(root,"diff","--name-only",BASE,"HEAD").splitlines())!=ALLOWED:raise ValueError("unexpected source change")
        if observed_identity(root)!=approved:raise ValueError("source or build hash mismatch")
        if sha256(COMPAT/"px4_yaw_compat.patch")!=approved["patch_sha256"]:raise ValueError("archived source patch mismatch")
        header=(build/"src/lib/version/build_git_version.h").read_text()
        if f'"{approved["commit"]}"' not in header:raise ValueError("build version mismatch")
        if not os.access(build/"bin/px4",os.X_OK):raise ValueError("binary not executable")
        return {**approved,"tag":git(root,"describe","--tags","--always"),"custom_sitl":True,
            "binary":str(build/"bin/px4"),"rootfs":str(build/"etc"),"manifest_sha256":sha256(MANIFEST)}
    except (OSError,KeyError,ValueError,subprocess.CalledProcessError) as e:
        raise ValueError("PX4 compatibility identity rejected: "+str(e)) from e
