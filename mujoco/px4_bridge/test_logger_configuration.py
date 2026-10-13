"""Explicit instance counts and private runtime configuration; no PX4 launch."""
from pathlib import Path

import pytest

from .test_interfaces import module


def test_finite_topic_file_has_required_instance_zero_without_duplicates():
    logger=module("logger_configuration")
    text=logger.render_topics()
    names={row[0] for row in logger.LOGGER_TOPICS}
    audit=logger.audit_topics(text,names)
    assert audit["subscriptions"]==19
    assert audit["instances"]==[0]
    assert set(audit["topics"])>=set(logger.REQUIRED_TOPICS)
    assert all(len(line.split())==3 and line.split()[2]=="0" for line in text.splitlines())


@pytest.mark.parametrize("change", ["implicit_instance","duplicate","missing_required","unknown","too_long"])
def test_invalid_or_implicit_topic_configuration_cannot_reach_logger(change):
    logger=module("logger_configuration")
    text=logger.render_topics();names={row[0] for row in logger.LOGGER_TOPICS}
    if change=="implicit_instance":text=text.replace("vehicle_visual_odometry 0 0","vehicle_visual_odometry 0")
    if change=="duplicate":text+="vehicle_visual_odometry 0 0\n"
    if change=="missing_required":text=text.replace("vehicle_visual_odometry 0 0\n","")
    if change=="unknown":text+="uncompiled_topic 0 0\n"
    if change=="too_long":text+="x"*80+" 0 0\n"
    with pytest.raises(ValueError):logger.audit_topics(text,names)


def test_subscription_capacity_is_enforced_without_automatic_multi_instances():
    logger=module("logger_configuration")
    text=logger.render_topics()+"".join(f"extra_{i} 0 0\n" for i in range(237))
    names={line.split()[0] for line in text.splitlines()}
    with pytest.raises(ValueError,match="subscription"):
        logger.audit_topics(text,names)


def test_private_etc_copy_cannot_modify_original_px4_startup_files(tmp_path):
    logger=module("logger_configuration")
    source=tmp_path/"original_etc";source.mkdir()
    (source/"init.d-posix").mkdir();(source/"init.d-posix/rcS").write_text("unchanged startup\n")
    runtime=tmp_path/"runtime";runtime.mkdir()
    before=module("compatibility_identity").tree_hash(source)
    metadata=logger.prepare_private_rootfs(runtime,source)
    assert not (runtime/"etc").is_symlink()
    assert metadata["startup_tree_sha256"]==before
    assert logger.startup_tree_hash(runtime/"etc")==before
    assert module("compatibility_identity").tree_hash(source)==before
    assert not (source/"logging/logger_topics.txt").exists()
    assert (runtime/"etc/logging/logger_topics.txt").read_text()==logger.render_topics()


def test_existing_runtime_etc_symlink_is_rejected(tmp_path):
    logger=module("logger_configuration")
    source=tmp_path/"source";source.mkdir()
    runtime=tmp_path/"runtime";runtime.mkdir();(runtime/"etc").symlink_to(source,target_is_directory=True)
    with pytest.raises(ValueError):logger.prepare_private_rootfs(runtime,source)
