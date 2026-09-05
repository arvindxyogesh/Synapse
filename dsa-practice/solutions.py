"""Solutions for problems covered in dsa-practice/*.md"""


def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """Two Sum on a sorted array. See 01_two_pointers.md."""
    left, right = 0, len(nums) - 1
    while left < right:
        current = nums[left] + nums[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    raise ValueError("no two sum solution")


def max_area(height: list[int]) -> int:
    """Container With Most Water. See 01_two_pointers.md."""
    left, right = 0, len(height) - 1
    best = 0
    while left < right:
        width = right - left
        best = max(best, min(height[left], height[right]) * width)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return best
