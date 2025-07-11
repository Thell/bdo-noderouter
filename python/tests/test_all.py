import pytest
import nwsf_rust


def test_sum_as_string():
    assert nwsf_rust.sum_as_string(1, 1) == "2"
