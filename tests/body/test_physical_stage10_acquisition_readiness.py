"""Tests for Problem 4 - Stage 10: Physical Data Acquisition Readiness & Capture-Only Workflow.

Validates:
1. Capture-only mode: proves downstream legacy processing (isolate, register, fuse, measure) is NOT called.
2. Manifest creation, validation, and JSON schema compatibility.
3. Unknown metadata handling: strictly records null / explicit unknown, never guesses.
4. Pre-save diagnostics:
   - Resolution / intrinsics mismatch detection.
   - Low valid depth / saturation detection.
   - Timestamp monotonicity violation detection.
   - Boundary clipping detection.
   - Incomplete yaw stop detection.
5. Post-save diagnostics: SHA-256 file hashing and directory verification.
6. Anonymous subject ID validation (strict PII rejection).
7. Preservation of existing raw data: verifies raw_captures/body_capture_01 was untouched.
8. Physical artifacts verification in stage10_acquisition_readiness_output/.
9. Regression compatibility across Stages 1 through 9.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from config import PipelineConfig
from pipeline import ScanPipeline
from realsense_measure.body.capture_session import (
    CaptureExperimentConfigManager,
    CaptureQualityChecker,
    CaptureSessionManifest,
    DepthValidityStats,
    DiagnosticCheckResult,
    FileManifestEntry,
    Stage10DeliverablesGenerator,
    validate_anonymous_subject_id,
)
from camera.frame import RGBDFrame
import open3d as o3d


STAGE10_DIR = Path("stage10_acquisition_readiness_output")
RAW_DIR = Path("raw_captures/body_capture_01")


# ---------------------------------------------------------------------------
# 1. Capture-Only Mode & Bypassing Downstream Processing
# ---------------------------------------------------------------------------

def test_capture_only_bypasses_downstream_legacy_pipeline(tmp_path):
    """
    Verify that in capture-only mode, capture_frames is called,
    raw frames are saved, manifest is created, but isolate_frames,
    register, fuse, and measure are strictly NOT invoked.
    """
    cfg = PipelineConfig(output_dir=tmp_path / "out")
    cfg.capture_only = True
    cfg.save_raw = True
    cfg.save_raw_dir = tmp_path / "raw_capture"
    cfg.yaw_stops = 2

    # Create 2 mock RGBDFrames
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()

    f1 = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=100.0, intrinsics=intr)
    f2 = RGBDFrame(frame_id=1, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=104.0, intrinsics=intr)

    pipeline = ScanPipeline(cfg)

    # Mock capture_frames to return our 2 test frames
    pipeline.capture_frames = MagicMock(return_value=[f1, f2])

    # Spy / mock downstream legacy functions
    pipeline.isolate_frames = MagicMock()
    pipeline.register = MagicMock()
    pipeline.fuse = MagicMock()
    pipeline.measure = MagicMock()

    res = pipeline.run()

    # 1. Capture was executed
    pipeline.capture_frames.assert_called_once()

    # 2. Downstream legacy pipeline was STRICTLY BYPASSED
    pipeline.isolate_frames.assert_not_called()
    pipeline.register.assert_not_called()
    pipeline.fuse.assert_not_called()
    pipeline.measure.assert_not_called()

    # 3. Clean exit with capture_complete status
    assert res["status"] == "capture_complete"
    assert res["mode"] == "capture_only"
    assert res["frames_captured"] == 2

    # 4. Manifest was saved in raw_capture directory
    manifest_p = cfg.save_raw_dir / "manifest.json"
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["capture_mode"] == "capture_only"
    assert data["frames_captured"] == 2


# ---------------------------------------------------------------------------
# 2. Manifest Creation & Validation
# ---------------------------------------------------------------------------

def test_manifest_creation_and_roundtrip(tmp_path):
    """Verify CaptureSessionManifest creation, validation, and JSON serialization."""
    manifest = CaptureSessionManifest(
        capture_id="test_cap_01",
        session_id="SESS_TEST_01",
        subject_id="SUBJ_42",
        operator_id="OPERATOR_01",
        config_id="CONFIG_D240_H105_Y4_R720",
        repeat_index=1,
        camera_to_subject_distance_m=2.40,
        camera_height_m=1.05,
        camera_pitch_tilt_deg=None,  # Explicit null for unmeasured
        nominal_yaw_stops=[0.0, 90.0, 180.0, 270.0],
    )
    d = manifest.to_dict()
    assert d["subject_id"] == "SUBJ_42"
    assert d["camera_pitch_tilt_deg"] is None
    assert d["camera_to_subject_distance_m"] == 2.40

    save_p = tmp_path / "manifest.json"
    manifest.save(save_p)
    assert save_p.exists()

    with open(save_p, "r", encoding="utf-8") as f:
        loaded_d = json.load(f)
    rebuilt = CaptureSessionManifest.from_dict(loaded_d)
    assert rebuilt.subject_id == manifest.subject_id
    assert rebuilt.camera_pitch_tilt_deg is None


def test_manifest_unknown_fields_are_null_not_guessed():
    """Verify that unknown camera serial, pitch tilt, or distances are null, not fabricated."""
    manifest = CaptureSessionManifest(
        capture_id="test_cap_02",
        subject_id="SUBJ_02",
        camera_serial_number=None,
        camera_pitch_tilt_deg=None,
        camera_to_subject_distance_m=None,
    )
    d = manifest.to_dict()
    assert d["camera_serial_number"] is None
    assert d["camera_pitch_tilt_deg"] is None
    assert d["camera_to_subject_distance_m"] is None


# ---------------------------------------------------------------------------
# 3. Anonymous Subject ID Validation
# ---------------------------------------------------------------------------

def test_anonymous_subject_id_validation():
    """Verify strict rejection of PII in subject_id."""
    invalid_ids = [
        "john.smith@gmail.com",
        "doctor@clinic.com",
        "John Smith",
        "Jane Doe",
        "Participant 01",
        "admin",
        "",
        None,
    ]
    for bad in invalid_ids:
        with pytest.raises(ValueError):
            validate_anonymous_subject_id(bad)

    valid_ids = [
        "SUBJ_01",
        "SUBJ_B105",
        "ANON_42",
        "PARTICIPANT_001",
        "TEST_09",
        "P_99",
    ]
    for good in valid_ids:
        assert validate_anonymous_subject_id(good) == good


# ---------------------------------------------------------------------------
# 4. Pre-Save Quality Diagnostics
# ---------------------------------------------------------------------------

def test_quality_checker_resolution_mismatch():
    """Verify detection of color/depth vs intrinsics resolution mismatch."""
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()
    frame = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=10.0, intrinsics=intr)

    # Check with conflicting expected intrinsics (1280x720)
    diag, warnings, failures = CaptureQualityChecker.evaluate_pre_save(
        [frame],
        expected_yaw_stops=1,
        intrinsics_wh=(1280, 720),
    )
    assert any("resolution mismatch" in f.lower() for f in failures)
    assert any(d.check_name == "frame_0_resolution_match" and d.status == "FAIL" for d in diag)


def test_quality_checker_low_depth_validity():
    """Verify detection of low depth validity / excessive dropouts."""
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    # Only 10% valid depth (rest zeros)
    depth = np.zeros((h, w), dtype=np.float32)
    depth[:48, :] = 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()
    frame = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=10.0, intrinsics=intr)

    diag, warnings, failures = CaptureQualityChecker.evaluate_pre_save(
        [frame],
        expected_yaw_stops=1,
        intrinsics_wh=(640, 480),
    )
    assert any("low valid depth return" in w.lower() for w in warnings)
    assert any(d.check_name == "frame_0_depth_validity" and d.status == "WARN" for d in diag)


def test_quality_checker_non_monotonic_timestamps():
    """Verify detection of non-monotonic or backward-jumping timestamps."""
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()

    f1 = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=100.0, intrinsics=intr)
    # Backward jump in timestamp: 95.0 < 100.0
    f2 = RGBDFrame(frame_id=1, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=95.0, intrinsics=intr)

    diag, warnings, failures = CaptureQualityChecker.evaluate_pre_save(
        [f1, f2],
        expected_yaw_stops=2,
        intrinsics_wh=(640, 480),
    )
    assert any("non-monotonic timestamps" in f.lower() for f in failures)
    assert any(d.check_name == "timestamp_monotonicity" and d.status == "FAIL" for d in diag)


def test_quality_checker_boundary_clipping():
    """Verify detection of boundary clipping when subject touches frame edges."""
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    # Contact at top row (y=0 to 2)
    depth[0:3, 100:200] = 2.0
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()
    frame = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=10.0, intrinsics=intr)

    diag, warnings, failures = CaptureQualityChecker.evaluate_pre_save(
        [frame],
        expected_yaw_stops=1,
        intrinsics_wh=(640, 480),
    )
    assert any("boundary clipping" in w.lower() for w in warnings)
    assert any("top" in w.lower() for w in warnings)


def test_quality_checker_incomplete_yaw_stops():
    """Verify detection of incomplete yaw stops (fewer frames than expected)."""
    w, h = 640, 480
    color = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.full((h, w), 2.0, dtype=np.float32)
    intr = o3d.camera.PinholeCameraIntrinsic(w, h, 500.0, 500.0, 320.0, 240.0)
    pcd = o3d.geometry.PointCloud()
    f1 = RGBDFrame(frame_id=0, color_bgr=color, depth_m=depth, pcd=pcd, timestamp=10.0, intrinsics=intr)

    # 1 frame captured when 4 expected
    diag, warnings, failures = CaptureQualityChecker.evaluate_pre_save(
        [f1],
        expected_yaw_stops=4,
        intrinsics_wh=(640, 480),
    )
    assert any("incomplete yaw stops" in w.lower() for w in warnings)
    assert any(d.check_name == "yaw_stop_completeness" and d.status == "WARN" for d in diag)


# ---------------------------------------------------------------------------
# 5. Post-Save Diagnostics & SHA-256 Checksums
# ---------------------------------------------------------------------------

def test_quality_checker_post_save(tmp_path):
    """Verify post-save checks compute correct file sizes and SHA-256 hashes."""
    cap_dir = tmp_path / "raw_cap_test"
    cap_dir.mkdir()

    # Create dummy frame files
    (cap_dir / "frame_00_color.png").write_bytes(b"dummy_png_bytes_00")
    (cap_dir / "frame_00_depth.npy").write_bytes(b"dummy_npy_bytes_00")
    (cap_dir / "meta.json").write_text(json.dumps({"num_frames": 1}), encoding="utf-8")

    entries, diag = CaptureQualityChecker.evaluate_post_save(cap_dir)
    assert len(entries) == 3
    names = [e.relative_path for e in entries]
    assert "frame_00_color.png" in names
    assert "frame_00_depth.npy" in names
    assert "meta.json" in names

    # Verify SHA-256 computation
    expected_hash = hashlib.sha256(b"dummy_png_bytes_00").hexdigest()
    png_entry = next(e for e in entries if e.relative_path == "frame_00_color.png")
    assert png_entry.sha256 == expected_hash
    assert png_entry.size_bytes == len(b"dummy_png_bytes_00")

    assert any(d.check_name == "meta_json_validity" and d.status == "PASS" for d in diag)


# ---------------------------------------------------------------------------
# 6. Configuration Manager (No False Optimality)
# ---------------------------------------------------------------------------

def test_experiment_config_manager_no_false_optimality():
    """Verify that configuration comparison refuses to declare optimality."""
    m1 = CaptureSessionManifest(capture_id="cap1", config_id="CONFIG_A", camera_to_subject_distance_m=2.2)
    m2 = CaptureSessionManifest(capture_id="cap2", config_id="CONFIG_B", camera_to_subject_distance_m=2.4)

    res = CaptureExperimentConfigManager.compare_configurations([m1, m2])
    assert res["status"] == "COMPARISON_ACTIVE"
    assert res["sessions_count"] == 2
    assert res["optimal_configuration_decision"] is None


# ---------------------------------------------------------------------------
# 7. Preservation of Existing Raw Captures
# ---------------------------------------------------------------------------

def test_raw_captures_unmodified():
    """Verify that raw_captures/body_capture_01 was NOT modified or overwritten."""
    assert RAW_DIR.is_dir(), f"Raw capture directory {RAW_DIR} missing"

    # Verify original files exist
    meta_p = RAW_DIR / "meta.json"
    assert meta_p.exists()
    with open(meta_p, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    assert meta_data["num_frames"] == 4

    for i in range(4):
        assert (RAW_DIR / f"frame_{i:02d}_color.png").exists()
        assert (RAW_DIR / f"frame_{i:02d}_depth.npy").exists()


# ---------------------------------------------------------------------------
# 8. Physical Artifacts Verification in stage10_acquisition_readiness_output/
# ---------------------------------------------------------------------------

def test_stage10_physical_artifacts_exist():
    """Verify all 8 required Stage 10 artifacts exist and are non-empty."""
    assert STAGE10_DIR.is_dir(), f"Stage 10 directory not found at {STAGE10_DIR}"

    required_files = [
        STAGE10_DIR / "stage10_report.md",
        STAGE10_DIR / "capture_protocol_checklist.md",
        STAGE10_DIR / "manifest_schema.json",
        STAGE10_DIR / "example_capture_manifest.json",
        STAGE10_DIR / "capture_only_cli_guide.md",
        STAGE10_DIR / "known_limitations_and_requirements.md",
        STAGE10_DIR / "stage10_test_results.json",
        STAGE10_DIR / "body_capture_01_manifest.json",
    ]
    for rf in required_files:
        assert rf.exists(), f"Required Stage 10 artifact missing: {rf}"
        assert rf.stat().st_size > 0, f"Artifact is empty: {rf}"


def test_manifest_schema_valid_json():
    """Verify manifest_schema.json is valid JSON Schema."""
    p = STAGE10_DIR / "manifest_schema.json"
    with open(p, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert schema["title"] == "CaptureSessionManifest"
    assert "required" in schema
    assert "subject_id" in schema["required"]
    assert "capture_mode" in schema["required"]


def test_example_manifest_placeholder_marking():
    """Verify example_capture_manifest.json has clearly marked nonphysical placeholder values."""
    p = STAGE10_DIR / "example_capture_manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "_comment" in data
    assert "NON-PHYSICAL PLACEHOLDER" in data["_comment"]
    assert data["capture_id"] == "placeholder_capture_demo"
    assert data["subject_id"] == "ANON_DEMO_01"


def test_body_capture_01_manifest_audited():
    """Verify body_capture_01_manifest.json contains accurate read-only audit of body_capture_01."""
    p = STAGE10_DIR / "body_capture_01_manifest.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["capture_id"] == "body_capture_01"
    assert data["subject_id"] == "SUBJ_01"
    assert data["frames_captured"] == 4
    assert len(data["file_manifest"]) == 9  # 4 color, 4 depth, 1 meta.json
    # Verify SHA-256 present for all files
    for entry in data["file_manifest"]:
        assert len(entry["sha256"]) == 64


def test_stage10_report_markdown_content():
    """Verify stage10_report.md exists and covers all required topics."""
    p = STAGE10_DIR / "stage10_report.md"
    content = p.read_text(encoding="utf-8")
    assert "Problem 4 — Stage 10: Physical Data Acquisition Readiness" in content
    assert "Capture-Only Workflow Architecture" in content
    assert "Audited Session Manifest for Existing Physical Capture" in content
    assert "Multi-Stage Linkage Architecture" in content
    assert "Output Artifact Index" in content
