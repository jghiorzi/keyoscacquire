# -*- coding: utf-8 -*-

import pytest

from pytest import approx

from keyoscacquire.dataprocessing import (
    process_data,
    _process_data_ascii,
    _process_data_binary,
)


@pytest.mark.parametrize("data_set", ["WORD", "BYTE", "ASCII"], indirect=True)
def test_process_data(data_set):
    data = data_set
    proc = process_data(data.raw, data.meta, data.fmt)
    for p, wp in zip(proc, data.processed):
        assert p == approx(wp)


def test_process_data_wrong_fmt():
    # Wrong wavformat
    with pytest.raises(ValueError):
        process_data([], [], "something")