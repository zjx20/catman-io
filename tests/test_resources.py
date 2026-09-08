import numpy as np

from training.wakeword.resources import parse_npy_header


def test_parse_npy_header_v1(tmp_path):
    arr = np.zeros((3, 16, 96), dtype=np.float16)
    path = tmp_path / "a.npy"
    np.save(path, arr)
    head = path.read_bytes()[:256]
    offset, header = parse_npy_header(head)
    assert header["shape"] == (3, 16, 96)
    assert header["descr"] == "<f2"
    assert offset + arr.nbytes == path.stat().st_size
