from solutions import max_area, two_sum_sorted


def test_two_sum_sorted_basic():
    assert two_sum_sorted([2, 7, 11, 15], 9) == [0, 1]


def test_two_sum_sorted_not_first_pair():
    assert two_sum_sorted([1, 2, 3, 4, 6], 10) == [3, 4]


def test_two_sum_sorted_negatives():
    assert two_sum_sorted([-4, -1, 0, 3, 10], 3) == [2, 3]  # 0 + 3 == 3


def test_two_sum_sorted_no_solution_raises():
    try:
        two_sum_sorted([1, 2, 3], 100)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_max_area_example():
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49


def test_max_area_two_elements():
    assert max_area([1, 1]) == 1


def test_max_area_increasing_heights():
    assert max_area([1, 2, 3, 4, 5]) == 6  # indices 0 and 4: min(1,5)*4


def test_max_area_all_equal():
    assert max_area([4, 4, 4, 4]) == 12  # widest pair, indices 0 and 3
