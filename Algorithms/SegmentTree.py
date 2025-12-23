import operator


class SegmentTree(object):
    def __init__(self, capacity: int, operation: operator, default: float):
        # Ensure capacity is positive and a power of two. (& is bitwise AND operator)
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._operation = operation
        self._value = [default for _ in range(2 * capacity)]

    def _reduce_helper(self, start: int, end: int, node: int, node_start: int, node_end: int):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start: int = 0, end: int = None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1

        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx: int, value: float):
        """
        Set index of leaf and iteratively update parent node to operator on its children.
        """
        idx += self._capacity
        self._value[idx] = value
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1],
            )
            idx //= 2

    def __getitem__(self, idx: int):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            default=0.0
        )

    def sum(self, start: int = 0, end: int = None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefix_sum_idx(self, prefix_sum: int) -> int:
        """
        Finds highest index `i` such that sum(arr[0],arr[1],...,arr[i-1]) < prefixsum

        If array values are probabilities, allows for sampling indices according priscrete probability efficiency.
        """
        assert 0 <= prefix_sum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            default=float('inf')
        )

    def min(self, start: int = 0, end: int = None):
        """
        Returns min(arr[start], ..., arr[end])
        """
        return super(MinSegmentTree, self).reduce(start, end)
