# Two Pointers

## Concept

When the input is sorted (or can be treated as such), use two indices moving
toward each other (or both forward) instead of checking all O(n^2) pairs.
Moving one pointer changes the comparison monotonically, which is what lets
you discard part of the search space on every step, collapsing many
brute-force problems from O(n^2) to O(n).

```python
left, right = 0, len(arr) - 1
while left < right:
    if condition_met(arr[left], arr[right]):
        # do something
        left += 1
        right -= 1
    elif need_bigger_sum:
        left += 1
    else:
        right -= 1
```

## Problem 1: Two Sum (sorted array)

Given a sorted array `nums` and a target value `target`, return the indices
of the two numbers that add up to `target`. Exactly one solution exists;
you can't use the same element twice.

Example: `nums = [2, 7, 11, 15]`, `target = 9` -> `[0, 1]`

### Solution

```python
def two_sum_sorted(nums: list[int], target: int) -> list[int]:
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
```

**Why it works:** because `nums` is sorted, moving `left` forward only
increases the sum and moving `right` backward only decreases it. That
monotonic behavior is the precondition for two pointers.

**Complexity:** O(n) time, O(1) extra space.

**Contrast with hash-map approach** (works on unsorted input too): iterate
once, and for each `num` check whether `target - num` has already been
seen; if not, add `num` to the seen-set. Also O(n) time, but O(n) space
instead of O(1) -- the price paid for not requiring sorted input.
