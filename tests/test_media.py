import numpy as np
from PIL import Image
import pytest

import engine
from media import box_masks, parse_box, read_clip, save_video


@pytest.fixture
def video(tmp_path):
    frames = [Image.new("RGB", (160, 96), (i * 5, 40, 80)) for i in range(25)]
    return save_video(frames, tmp_path / "source.mp4", fps=24)


def test_resampling_preserves_time_and_vace_frame_count(video):
    frames = read_clip(video, (160, 96), 33)
    assert len(frames) == 17
    assert np.asarray(frames[-1])[40, 80, 0] > 100
    assert np.asarray(frames[0])[40, 80, 0] < 10
    assert (len(frames) - 1) % 4 == 0


def test_mask_moves_between_supplied_endpoints():
    masks = box_masks((160, 96), 17, "0,0,0.25,0.5", "0.75,0.5,1,1")
    assert masks[0].getbbox() == (0, 0, 40, 48)
    assert masks[-1].getbbox() == (120, 48, 160, 96)
    assert masks[8].getbbox() == (60, 24, 100, 72)
    assert np.asarray(masks[8])[30, 70] == 255
    assert np.asarray(masks[8])[0, 0] == 0


@pytest.mark.parametrize("box", ["1,0,0,1", "0,0,2,1", "nan,0,1,1", "0,1", "hello", "0,0,0,1"])
def test_invalid_regions_fail_before_generation(box):
    with pytest.raises(ValueError):
        parse_box(box)


def test_mp4_roundtrip_contains_actual_frames(tmp_path):
    frames = [Image.new("RGB", (160, 96), (20 + i * 5, 60, 90)) for i in range(17)]
    path = save_video(frames, tmp_path / "roundtrip.mp4")
    result = read_clip(path, (160, 96), 17)
    assert len(result) == 17
    assert np.asarray(result[-1])[30, 60, 0] > np.asarray(result[0])[30, 60, 0] + 60


def test_no_gpu_never_creates_fake_result(video, tmp_path, monkeypatch):
    reference = tmp_path / "ref.png"
    Image.new("RGB", (160, 96), "blue").save(reference)
    def unavailable():
        raise RuntimeError("GPU indisponível")
    monkeypatch.setattr(engine, "require_gpu", unavailable)
    with pytest.raises(RuntimeError, match="GPU indisponível"):
        engine.generate(video, reference, "swap", "a blue box", size=(160, 96),
                        output_root=tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_invalid_file_rejected(tmp_path):
    path = tmp_path / "bad.mp4"
    path.write_text("not a movie")
    with pytest.raises(ValueError):
        read_clip(path)
