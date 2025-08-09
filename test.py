from typing import List

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0:
            return 0

        # Prefix ORs
        pref = [0] * n
        cur = 0
        for i in range(n):
            cur |= nums[i]
            pref[i] = cur

        # Suffix ORs
        suff = [0] * n
        cur = 0
        for i in range(n - 1, -1, -1):
            cur |= nums[i]
            suff[i] = cur

        # Try placing all k doublings on each index i
        best = 0
        for i in range(n):
            excl = (pref[i - 1] if i > 0 else 0) | (suff[i + 1] if i < n - 1 else 0)
            cand = (nums[i] << k) | excl
            if cand > best:
                best = cand

        return best

solu = Solution()
print(solu.maximumOr([12, 9], 1))