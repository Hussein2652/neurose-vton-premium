from pathlib import Path

from neurose_vton.pipeline import TryOnPipeline, TryOnConfig


def test_person_stage_writes_trace(tmp_path: Path):
    # Create dummy inputs
    person = tmp_path / "person.png"
    garment = tmp_path / "garment.png"
    person.write_bytes(b"")
    garment.write_bytes(b"")

    cfg = TryOnConfig(steps=1, save_intermediates=True, seed=123)
    result = TryOnPipeline(cfg).run(person, garment)
    assert result.trace_dir is not None
    # Person artifacts should exist
    person_dir = result.trace_dir / "person"
    for name in [
        'face_detection.json','face_landmarks.json','face_embedding.json',
        'pose.json','segmentation.png','hair_mask.png','arms_mask.png','hands_mask.png',
        'depth.png','normals.png','body_mesh.obj','light_sh.json'
    ]:
        assert (person_dir / name).exists()
