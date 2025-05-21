import pytest

@pytest.mark.mpi(min_size=2)
def test_multiple_prints():
    print("aaaa")
    assert True

@pytest.mark.mpi_skip()
def test_single_print():
    print("bbbb")
    assert True
