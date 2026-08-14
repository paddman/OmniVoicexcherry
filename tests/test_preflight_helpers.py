import pytest

from omnivoice.utils.preflight import parse_gpu_ids


def test_parse_gpu_ids():
    assert parse_gpu_ids("cpu") == (True, [])
    assert parse_gpu_ids("0,2") == (False, [0, 2])


def test_parse_gpu_ids_rejects_bad_values():
    with pytest.raises(ValueError):
        parse_gpu_ids("0,cuda")
    with pytest.raises(ValueError):
        parse_gpu_ids("1,1")
