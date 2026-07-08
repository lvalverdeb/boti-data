from __future__ import annotations

import pytest

import boti_data
import boti_data.distributed


def test_top_level_dask_resolved_as_direct_reexport():
    assert boti_data.DaskSession is boti_data.distributed.DaskSession


def test_top_level_dask_session_factory_resolved():
    assert callable(boti_data.dask_session)


def test_unknown_top_level_symbol_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = boti_data.__definitely_not_a_symbol__

