import pytest

from app.config import check_python_version


@pytest.mark.parametrize("version", [(3, 11), (3, 12), (3, 13)])
def test_supported_versions_pass(version):
    check_python_version(version)


@pytest.mark.parametrize("version", [(3, 10), (3, 14), (3, 15)])
def test_unsupported_versions_fail_fast(version):
    with pytest.raises(RuntimeError, match="not supported"):
        check_python_version(version)
