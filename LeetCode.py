# 3. Longest Substring Without Repeating Characters
class Solution:
	def lengthOfLongestSubstring(self, s: str) -> int:
					max = 0
		res_str = ''
		length = len(s)
		for i in range(length):
			if s[i] not in res_str:
				res_str += s[i]
				if len(res_str) > max:
					max = len(res_str)
			else:
				pos = res_str.index(s[i])
				res_str = res_str[pos+1:] + s[i]

		return max

# 4. Median of Two Sorted Arrays
class Solution:
	def findMedianSortedArrays(self, nums1, nums2) -> float:
		l3 = sorted(nums1 + nums2)
		return l3[len(l3) // 2] if len(l3) % 2 == 1 else (l3[len(l3) // 2 - 1] + l3[len(l3) // 2]) / 2

# 7. Reverse Integer
class Solution:
	def reverse(self, x: int) -> int:
		flag = 1 if x > 0 else -1
		x = x if x > 0 else -x
		x = int(str(x)[::-1])
		if x > (pow(2, 31) - 1) or x < -pow(2, 31):
			return 0
		return flag*x

# 9. Palindrome Number
class Solution:
	def isPalindrome(self, x: int) -> bool:
		if str(x)[::-1] == str(x):
			return True
		return False

# 13. Roman to Integer
class Solution:
	def romanToInt(self, s: str) -> int:
		special = {"IV": 4, "IX": 9, "XL": 40, "XC": 90, "CD": 400, "CM": 900}
		normal = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
		res = 0
		for ch in special.keys():
			if ch in s:
				s.replace(ch, "")
				res += special[ch]
		for ch in s:
			res += normal[ch]
		return res

# 14. Longest Common Prefix
class Solution:
	def longestCommonPrefix(self, strs) -> str:
		res = ""
		str_num = len(strs)
		if str_num == 0:
			return res
		length = [len(str) for str in strs]
			len = min(length)
		for i in range(	len):
			for j in range(str_num - 1):
				if strs[j][i] != strs[j + 1][i]:
					return res
			res += strs[0][i]
		return res

# 14. Longest Common Prefix, set niubility
class Solution:
	def longestCommonPrefix(self, strs) -> str:
		res = ""
		strs = zip(*strs)
		for x in strs:
			sx = set(x)
			if len(sx) == 1:
				res += list(sx)[0]
			else:
				return res
		return res

# 20. Valid Parentheses
class Solution:
	def isValid(self, s: str) -> bool:
		left = ['(', '[', '{']
		right = [')', ']', '}']
		stack = []
		for ch in s:
			if ch in left:
				stack.append(ch)
			else:
				if len(stack) == 0 or stack[-1] != left[right.index(ch)]:
					return False
				stack.pop()

		if len(stack) == 0:
			return True
		return False

# 20. Valid Parentheses, easy to understand
class Solution:
	def isValid(self, s: str) -> bool:
		length = len(s)
		while (length):
			s = s.replace('()', '')
			s = s.replace('[]', '')
			s = s.replace('{}', '')
			if len(s) == length:
				return False
			length = len(s)
		return True

# 21. Merge Two Sorted Lists
# Definition for singly-linked list.
class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None

class Solution:
	def mergeTwoLists(self, l1: ListNode, l2e) -> ListNode:
		if l1 is None: return l2
		if l2 is None: return l1
		if l1 is None and l2 is None: return None

		head = ListNode(0)
		cur = head

		while l1 and l2:
			if l1.val < l2.val:
				cur.next = l1
				l1 = l1.next
			else:
				cur.next = l2
				l2 = l2.next
			cur = cur.next

		cur.next = l1 or l2
		return head.next

# 26. Remove Duplicates from Sorted Array
class Solution:
	def removeDuplicates(self, nums) -> int:
		length = 0
		for num in nums:
			i = 0
			while i < length:
				if nums[i] == num:
					break
				i += 1
			if i == length:
				nums[length] = num
				length += 1

		return length

# 26. Remove Duplicates from Sorted Array
# !!!!!!! NOTE:SORTED
class Solution:
	def removeDuplicates(self, nums) -> int:
		sets = list(set(nums))
		sets.sort()
		length = len(sets)
		for i in range(length):
			nums[i] = sets[i]
		return length

# 26. Remove Duplicates from Sorted Array
# !!!!!!! NOTE:SORTED
class Solution:
	def removeDuplicates(self, nums) -> int:
		if len(nums) <= 1:
			return len(nums)
		length = 0
		for i in range(1, len(nums)):
			if nums[i] != nums[length]:
				length += 1
				nums[length] = nums[i]
		return length + 1


# 27. Remove Element
class Solution:
	def removeElement(self, nums, val: int) -> int:
		length = len(nums)
		k = 0
		for i in range(length):
			if nums[i] != val:
				nums[k] = nums[i]
				k += 1
		return k

# 5. Longest Palindromic Substring
class Solution:
	def longestPalindrome(self, s: str) -> str:
		length = len(s)
		l = length
		if length <= 1: return s
		while l > 1:
			for i in range(length - l + 1):
				if s[i:i+l] == s[i:i+l][::-1]:
					return s[i:i+l]
			l -= 1
		return s[0]

# 28. Implement strStr()
class Solution:
	def strStr(self, haystack: str, needle: str) -> int:
		length_haystack = len(haystack)
		length_needle = len(needle)
		if length_needle > length_haystack: return -1
		for i in range(length_haystack - length_needle + 1):
			if haystack[i:length_needle + i] == needle:
				return i
		return -1

# 6. ZigZag Conversion
class Solution:
	def convert(self, s: str, numRows: int) -> str:
		length = len(s)
		res = ''
		if length <= numRows or numRows == 1:
			return s
		nums = []
		for i in range(1, numRows):
			nums.append((numRows - i) * 2)
		nums.append((numRows - 1) * 2)

		for i in range(numRows):
			j = i
			flag = 0
			while j < length:
				res += s[j]
				if nums[i] == (numRows - 1) * 2:
					j += nums[i]
				elif flag:
					j += (numRows - 1) * 2 - nums[i]
					flag = 0
				else:
					j += nums[i]
					flag = 1
		return res

# 6. ZigZag Conversion # niubility
class Solution:
	def convert(self, s: str, numRows: int) -> str:
		if numRows == 1 or len(s) <= numRows:
			return s

		L = [''] * numRows
		index, step = 0, 1

		for ch in s:
			L[index] += ch
			if index == 0:
				step = 1
			elif index == numRows - 1:
				step = -1
			index += step

		return ''.join(L)

# 35. Search Insert Position
class Solution:
	def searchInsert(self, nums, target: int) -> int:
		if len(nums) == 0 or nums[len(nums) - 1] < target:
			return len(nums)
		for i in range(len(nums)):
			if nums[i] >= target:
				return i

# 8. String to Integer (atoi)
class Solution:
	def myAtoi(self, str: str) -> int:
		match = re.match(r'^ *([+-]{0,1}\d+)', str)
		if not match:
			return 0
		n = int(match[1])

		if n < -(2 ** 31): return -(2 ** 31)
		if n >= 2 ** 31: return 2 ** 31 - 1
		return n

# 53. Maximum Subarray
class Solution:
	def maxSubArray(self, nums) -> int:
		for i in range(1, len(nums)):
			if nums[i - 1] > 0:
				nums[i] += nums[i - 1]
		return max(nums)

# 11. Container With Most Water
class Solution:
	def maxArea(self, height) -> int:
		Max = -1
		l = 0
		r = len(height) - 1
		while l < r:
			Max = max(Max, min(height[r], height[l]) * (r - l))
			if height[l] < height[r]:
				l += 1
			else:
				r -= 1
		return Max

# 12. Integer to Roman
class Solution:
	def intToRoman(self, num: int) -> str:
		value = {
			0: ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'],
			1: ['', 'X', 'XX', 'XXX', 'XL', 'L', 'LX', 'LXX', 'LXXX', 'XC'],
			2: ['', 'C', 'CC', 'CCC', 'CD', 'D', 'DC', 'DCC', 'DCCC', 'CM'],
			3: ['', 'M', 'MM', 'MMM']
		}
		i = 0
		res = ''
		while num:
			res = value[i][num % 10] + res
			num //= 10
			i += 1

		return res

# 58. Length of Last Word
class Solution:
	def lengthOfLastWord(self, s: str) -> int:
		import re
		res = re.findall(r'\w+', s)
		return len(res[-1]) if len(res) > 0 else 0

# 66. Plus One 65.75%
class Solution:
	def plusOne(self, digits):
		num = 0
		for i in range(len(digits)):
			num = num * 10 + digits[i]
		num += 1
		res = []
		while num:
			res.insert(0, num % 10)
			num = num // 10
		return res

# 70. Climbing Stairs 67.05%
class Solution:
	def climbStairs(self, n: int) -> int:
		res = [1, 2]
		i = n
		while i > 2:
			res.append(res[-1] + res[-2])
			i -= 1
		return res[n - 1]

# 15. 3Sum 借鉴的别人的很优秀的方法，牛皮 89.81%
class Solution:
	def threeSum(self, nums):
		res = []
		nums.sort()
		length = len(nums)
		for i in range(length - 2):
			if nums[i] > 0: break
			if i > 0 and nums[i] == nums[i - 1]: continue
			l, r = i + 1, length - 1
			while l < r:
				total = nums[i] + nums[l] + nums[r]
				if total > 0:
					r -= 1
				elif total < 0:
					l += 1
				else:
					res.append([nums[i], nums[l], nums[r]])
					while l < r and nums[l] == nums[l + 1]:
						l += 1
					while l < r and nums[r] == nums[r - 1]:
						r -= 1
					l += 1
					r -= 1
		return res

# 16. 3Sum Closest 78.09%
class Solution:
	def threeSumClosest(self, nums, target: int) -> int:
		res = nums[0] + nums[1] + nums[2] - target
		nums.sort()
		length = len(nums)
		for i in range(length - 2):
			if i > 0 and nums[i] == nums[i - 1]: continue
			l, r = i + 1, length - 1
			while l < r:
				diff = nums[i] + nums[l] + nums[r] - target
				if diff == 0: return target
				if abs(res) > abs(diff): res = diff
				elif diff > 0: r -= 1
				else: l += 1
		return res + target

# 69. Sqrt(x)
class Solution:
	def mySqrt(self, x: int) -> int:
		if x < 2: return x
		l, r = 1, x // 2
		while l < r:
			m = (l + r) // 2
			m_2 = m ** 2
			if m_2 < x:
				l = m + 1
			else:
				r = m
		return l if l ** 2 <= x else l - 1

# 17. Letter Combinations of a Phone Number
class Solution:
	def letterCombinations(self, digits: str):
		phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
			'6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
		if not digits:
			return []

		def helper(digits):
			if not digits:
				return ['']
			return [c + c_ for c in phone[digits[0]]
						for c_ in helper(digits[1:])]

		return helper(digits)

# 17. Letter Combinations of a Phone Number
class Solution:
	def letterCombinations(self, digits: str):
		phone = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
			'6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
		res = []
		if not digits:
			return []

		def helper(res_, digits):
			if not digits:
				res.append(res_)
			else:
				for ch in phone[digits[0]]:
					helper(res_ + ch, digits[1:])

		helper('', digits)
		return res

# 83. Remove Duplicates from Sorted List 81.13%
# Definition for singly-linked list.
# class ListNode:
#	 def __init__(self, x):
#		self.val = x
#		self.next = None

class Solution:
	def deleteDuplicates(self, head):
		p = head
		while p and p.next:
			if p.val == p.next.val:
				while p.next and p.val == p.next.val:
					p.next = p.next.next
			p = p.next
		return head

# 46. Permutations 63.19%
class Solution:
	def permute(self, nums):
		res = []
		def helper(res_, nums):
			if len(nums) == 0:
				res.append(res_)
			else:
				for i in range(len(nums)):
					res__ = list(res_)
					res__.append(nums[i])
					helper(res__, nums[:i] + nums[i + 1:])
		helper([], nums)
		return res

# 38. Count and Say 77.51%
class Solution:
	def countAndSay(self, n: int) -> str:
		if n == 1:
			return '1'
		res = self.countAndSay(n - 1)
		res_ = ''
		length = len(res)
		i = 0
		while i < length:
			count = 1
			while i + 1 < length and res[i] == res[i + 1]:
				count += 1
				i += 1
			res_ = res_ + str(count) + res[i]
			i += 1
		return res_

# Definition for singly-linked list.
# class ListNode:
# 	def __init__(self, x):
# 		self.val = x
# 		self.next = None

# 19. Remove Nth Node From End of List 89.95%
class Solution:
	def removeNthFromEnd(self, head, n: int):
		length = 0
		p = head
		while p:
			length += 1
			p = p.next
		if length == n:
			return head.next

		p = head
		while length - n - 1 > 0:
			p = p.next
			n += 1
		p.next = p.next.next
		return head

# 88. Merge Sorted Array 70.51%
class Solution:
	def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
		"""
		Do not return anything, modify nums1 in-place instead.
		"""
		i = m + n - 1
		while i >= n:
			nums1[i] = nums1[i - n]
			i -= 1
		i = 0
		j = 0
		k = 0
		while i < m and j < n:
			if nums1[n + i] < nums2[j]:
				nums1[k] = nums1[n + i]
				i += 1
			else:
				nums1[k] = nums2[j]
				j += 1
			k += 1

		while j < n:
			nums1[k] = nums2[j]
			j += 1
			k += 1

# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None
# 100. Same Tree 75.97%
class Solution:
	def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
		if p is None and q is None: return True
		if p is None or q is None: return False
		return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# 118. Pascal's Triangle 73.01%
class Solution:
	def generate(self, numRows: int) -> List[List[int]]:
		res = []
		for i in range(1, numRows + 1):
			res_ = []
			for j in range(i):
				res_.append(1 if j == 0 or j == i - 1 else res[-1][j] + res[-1][j+1])
			res.append(res_)
		return res

# 18. 4Sum 18.30%
class Solution:
	def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
		nums.sort()
		length = len(nums)
		res = []
		for i in range(length - 3):
			if i > 0 and nums[i] == nums[i - 1]: continue
			if nums[i] + nums[i + 1] + nums[i + 2] > target and nums[i + 3] >= 0: break
			for j in range(i + 1, length - 2):
				if j > i + 1 and nums[j] == nums[	]: continue
				l = j + 1
				r = length - 1
				while l < r:
					if nums[i] + nums[j] + nums[l] + nums[r] < target:
						l += 1
					elif nums[i] + nums[j] + nums[l] + nums[r] > target:
						r -= 1
					else:
						res.append([nums[i], nums[j], nums[l], nums[r]])
						while l < r and nums[l] == nums[l + 1]: l += 1
						while l < r and nums[r] == nums[r - 1]: r -= 1
						l += 1
						r -= 1

		return res

# 136. Single Number 46.30%
class Solution:
	def singleNumber(self, nums: List[int]) -> int:
		res = 0
		for num in nums:
			res ^= num
		return res

# 36. Valid Sudoku 63.93%
class Solution:
	def isValidSudoku(self, board: List[List[str]]) -> bool:
		board_ = []
		for i in range(9):
			board_row = []
			board_column = []
			board_box = []
			for j in range(9):
				if board[i][j] != '.':
					board_row.append(board[i][j])
				if board[j][i] != '.':
					board_column.append(board[j][i])
				if board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3] != '.':
					board_box.append(board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3])
			board_.append(board_row)
			board_.append(board_column)
			board_.append(board_box)

		for _ in board_:
			if len(_) != len(set(_)):
				return False
		return True

# 141. Linked List Cycle
# Definition for singly-linked list.
# class ListNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.next = None

class Solution(object): # hash table 5.27%
	def hasCycle(self, head):
		"""
		:type head: ListNode
		:rtype: bool
		"""
		nodes = []
		while head is not None:
			if head in nodes:
				return True
			else:
				nodes.append(head)
			head = head.next
		return False

class Solution(object): # two pointer 70.05%
	def hasCycle(self, head):
		"""
		:type head: ListNode
		:rtype: bool
		"""
		if not head or not head.next: return False
		slow = head
		fast = head.next

		while slow != fast:
			if not fast or not fast.next: return False
			slow = slow.next
			fast = fast.next.next

		return True

# 24. Swap Nodes in Pairs
# Definition for singly-linked list.
# class ListNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.next = None

class Solution(object): # 71.44%
	def swapPairs(self, head):
		"""
		:type head: ListNode
		:rtype: ListNode
		"""
		if not head or not head.next: return head
		p = head
		q = p.next
		res = q
		while p and q:
			pre = p
			p.next = q.next
			q.next = p
			p = p.next
			q = p.next if p else None
			pre.next = q if q else p
		return res

class Solution(object): # recursion 71.44%
	def swapPairs(self, head):
		"""
		:type head: ListNode
		:rtype: ListNode
		"""
		if not head or not head.next: return head
		p = head.next
		head.next = self.swapPairs(p.next)
		p.next = head
		return p

# 67. Add Binary 71.41%
class Solution(object):
	def addBinary(self, a, b):
		"""
		:type a: str
		:type b: str
		:rtype: str
		"""
		def helper(res, a, b, c):
			if len(a) == 0 and len(b) == 0:
				return ('1' if c else '') + res
			c, d = divmod(c + (1 if len(a) and a[-1] == '1' else 0) + (1 if len(b) and b[-1] == '1' else 0), 2)
			return helper(('1' if d else '0') + res, a[:-1], b[:-1], c)
		res = helper('', a, b, 0)
		return res

# 29. Divide Two Integers 100.00%
class Solution(object):
	def divide(self, dividend, divisor):
		"""
		:type dividend: int
		:type divisor: int
		:rtype: int
		"""
		# overflow situation
		if dividend == -2147483648 and divisor == -1:
			return 2147483647
		res, rem = divmod(dividend, divisor)
		return res + 1 if res < 0 and rem else res

# 167. Two Sum II - Input array is sorted 80.69%
class Solution:
	def twoSum(self, numbers: List[int], target: int) -> List[int]:
		l = 0
		r = len(numbers) - 1
		while l < r:
			if numbers[l] + numbers[r] < target:
				while l < r and numbers[l + 1] == numbers[l]:
					l += 1
				l += 1
			elif numbers[l] + numbers[r] > target:
				while l < r and numbers[r - 1] == numbers[r]:
					r -= 1
				r -= 1
			else:
				return [l + 1, r + 1]

# 75. Sort Colors 65.04%
class Solution:
	def sortColors(self, nums: List[int]) -> None:
		"""
		Do not return anything, modify nums in-place instead.
		"""
		length = len(nums)
		times = [0, 0, 0]
		for i in range(length):
			times[nums[i]] += 1
		k = 0
		for i in range(3):
			for j in range(times[i]):
				nums[k] = i
				k += 1
class Solution: # 97.96% 双指针
	def sortColors(self, nums: List[int]) -> None:
		"""
		Do not return anything, modify nums in-place instead.
		"""
		i, l, r = 0, 0, len(nums) - 1
		while i <= r:
			if nums[i] == 0:
				nums[i], nums[l] = nums[l], nums[i]
				l += 1
			if nums[i] == 2:
				nums[i], nums[r] = nums[r], nums[i]
				r -= 1
				i -= 1
			i += 1

# 125. Valid Palindrome 99.95%
class Solution:
	def isPalindrome(self, s: str) -> bool:
		from string import punctuation
		for ch in punctuation:
			s = s.replace(ch, '')
		s = s.replace(' ', '')
		return s.lower() == s.lower()[::-1]

# 22. Generate Parentheses 57.81%
class Solution:
	def generateParenthesis(self, n: int) -> List[str]:
		if n == 1:
			return ['()']
		res = []
		for s in self.generateParenthesis(n - 1):
			for i in range(len(s)):
				res_ = s[:i] + '()' + s[i:]
				res.append(res_)
		return list(set(res))

# 278. First Bad Version 67.87%
# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
	def firstBadVersion(self, n):
		"""
		:type n: int
		:rtype: int
		"""
		if n < 2: return n
		l, r = 1, n
		while l < r:
			m = (l + r) // 2
			if isBadVersion(m):
				r = m
			else:
				l = m + 1
		return l

# 33. Search in Rotated Sorted Array 99.22%
class Solution:
	def search(self, nums: List[int], target: int) -> int:
		if len(nums) == 0:
			return -1

		l, r = 0, len(nums) - 1
		while l < r:
			m = (l + r) // 2
			if nums[m] == target:
				return m
			if nums[l] == target:
				return l
			if nums[r] == target:
				return r

			elif nums[m] < target:
				if nums[l] > target:
					l = m + 1
				else:
					if nums[m] >= nums[l]:
						l = m + 1
					else:
						r = m - 1
			else:
				if nums[l] < target:
					r = m - 1
				else:
					if nums[m] >= nums[l]:
						l = m + 1
					else:
						r = m - 1

		return l if nums[l] == target else -1

# 169. Majority Element
class Solution: # 5.06%
	def majorityElement(self, nums: List[int]) -> int:
		if len(nums) < 3:
			return nums[0]
		r = 1
		while r < len(nums):
			while r < len(nums) and nums[r] == nums[0]: r += 1
			if r < len(nums):
				nums.pop(0)
				r -= 1
				nums.pop(r)
		return nums[0]
class Solution: # 40.58%
	def majorityElement(self, nums: List[int]) -> int:
		if len(nums) < 3:
			return nums[0]
		counter = {}
		for num in nums:
			if num not in counter.keys():
				counter[num] = 1
			else:
				counter[num] += 1
		length = len(nums)
		for num in counter.keys():
			if counter[num] > length // 2:
				return num
class Solution: # 67.32%
	def majorityElement(self, nums: List[int]) -> int:
		if len(nums) < 3:
			return nums[0]
		counter = 0
		candidate = None
		for num in nums:
			if counter == 0:
				candidate = num
			counter += (1 if num == candidate else -1)
		return candidate

class Solution: # 99.51%
	def majorityElement(self, nums: List[int]) -> int:
		return sorted(nums)[len(nums)//2]

# 215. Kth Largest Element in an Array
class Solution: # without heap or hash 5.01%
	def findKthLargest(self, nums: List[int], k: int) -> int:
		length = len(nums)
		for i in range(k):
			for j in range(length - i - 1):
				if nums[j] > nums[j + 1]:
					nums[j], nums[j + 1] = nums[j + 1], nums[j]
		return nums[-k]
class Solution: # cheat 90.92%
	def findKthLargest(self, nums: List[int], k: int) -> int:
		return sorted(nums)[-k]

# 121. Best Time to Buy and Sell Stock
class Solution(object): # 35.90%
	def maxProfit(self, prices):
		"""
		:type prices: List[int]
		:rtype: int
		"""
		if len(prices) < 2: return 0
		min = prices[0]
		max = 0
		for price in prices:
			if price - min > max: max = price - min
			if price < min: min = price
		return max

# 62. Unique Paths
class Solution(object):
	def uniquePaths(self, m, n): # 14.98%
		"""
		:type m: int
		:type n: int
		:rtype: int
		"""
		if m == 1 or n == 1: return 1
		dp = [[1 for _ in range(m)] for _ in range(n)]
		for i in range(1, n):
			for j in range(1, m):
				dp[i][j] = dp[i - 1][j] + dp[i][	]
		return dp[n - 1][m - 1]
class Solution(object): # 76.75%
	def uniquePaths(self, m, n):
		"""
		:type m: int
		:type n: int
		:rtype: int
		"""
		return math.factorial(m+n-2)/math.factorial(m-1)/math.factorial(n-1)

# 401. Binary Watch
class Solution(object): # 54.13%
	def readBinaryWatch(self, num):
		"""
		:type num: int
		:rtype: List[str]
		"""
		return ['%d:%02d' % (h, m)
				for h in range(12) for m in range(60)
				if (bin(h) + bin(m)).count('1') == num]

# 104. Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # recursion 41.52%
	def maxDepth(self, root):
		"""
		:type root: TreeNode
		:rtype: int
		"""
		if not root: return 0
		return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
class Solution(object): # stack 41.52%
	def maxDepth(self, root):
		"""
		:type root: TreeNode
		:rtype: int
		"""
		if not root: return 0
		stack = []
		stack.append((1, root))
		depth = 0
		while stack != []:
			cur_depth, root = stack.pop()
			if root:
				depth = max(cur_depth, depth)
				stack.append((cur_depth + 1, root.left))
				stack.append((cur_depth + 1, root.right))
		return depth

# 1021. Remove Outermost Parentheses
class Solution(object): # 62.88%
	def removeOuterParentheses(self, S):
		"""
		:type S: str
		:rtype: str
		"""
		res, counter = [], 0
		for c in S:
			if c == '(' and counter > 0: res.append(c)
			if c == ')' and counter > 1: res.append(c)
			counter += 1 if c == '(' else -1

		return ''.join(res)

# 31. Next Permutation
class Solution(object):
	def nextPermutation(self, nums): # 58.26%
		"""
		:type nums: List[int]
		:rtype: None Do not return anything, modify nums in-place instead.
		"""
		j, l = len(nums) - 1, len(nums)
		while j:
			if nums[j] > nums[	]:
				break
			j -= 1
		if j:
			if nums[-1] > nums[	]:
				nums[-1], nums[	] = nums[	], nums[-1]
			else:
				for i in range(j, l):
					if nums[i] > nums[	] and nums[i + 1] <= nums[	]:
						nums[i], nums[	] = nums[	], nums[i]
			for i in range((l - j) // 2):
				nums[i + j], nums[l - i - 1] = nums[l - i - 1], nums[i + j]
		else:
			nums.sort()

# 198. House Robber
class Solution(object):
	def rob(self, nums): # 73.88%
		"""
		:type nums: List[int]
		:rtype: int
		"""
		length = len(nums)
		if length == 0: return 0
		if length == 1: return nums[0]
		if length == 2: return max(nums)
		cur_max = [nums[i] for i in range(length)]
		cur_max[1] = max(nums[0], nums[1])
		for i in range(2, length):
			cur_max[i] = max(cur_max[i - 1], cur_max[i - 2] + nums[i])
		return cur_max[-1]
class Solution(object):
	def rob(self, nums): # 73.88% 这个写得好简洁啊
		"""
		:type nums: List[int]
		:rtype: int
		"""
		pre = cur = 0
		for x in nums:
			cur, pre = max(cur, pre + x), cur
		return cur

#————————————————19/4/12———————————————————
# 64. Minimum Path Sum
class Solution(object):
	def minPathSum(self, grid): # 35.66%
		"""
		:type grid: List[List[int]]
		:rtype: int
		"""
		if grid == []: return 0
		h, w = len(grid), len(grid[0])
		for i in range(1, h):
			grid[i][0] += grid[i - 1][0]
		for j in range(1, w):
			grid[0][j] += grid[0][j - 1]

		for i in range(1, h):
			for j in range(1, w):
				grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])

		return grid[-1][-1]

# 303. Range Sum Query - Immutable
class NumArray(object): # 55.14%

	def __init__(self, nums):
		"""
		:type nums: List[int]
		"""
		l = len(nums)
		self.sum_list = [0] * (l + 1)
		for i in range(l):
			self.sum_list[i + 1] = self.sum_list[i] + nums[i]

	def sumRange(self, i, j):
		"""
		:type i: int
		:type j: int
		:rtype: int
		"""
		return self.sum_list[j + 1] - self.sum_list[i]

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

#————————————————19/4/13———————————————————
# 91. Decode Ways
class Solution(object):
	def numDecodings(self, s): # 感觉是对的，但是使用递归导致重复计算过多 进而超时
		"""
		:type s: str
		:rtype: int
		"""
		if not s or s[0] == '0': return 0
		illegal_strs = ['00', '30', '40', '50', '60', '70', '80', '90']
		for illegal_str in illegal_strs:
			if illegal_str in s: return 0

		def helper(s_):
			l = len(s_)
			if len(s_) < 2: return 1
			return helper(s_[1:]) if int(s_[:2]) > 26 else helper(s_[1:]) + helper(s_[2:])

		if '0' in s:
			s_ = s.split('0')
			res = 1
			for _ in s_:
				res *= helper(_[:-1])
			return res

		return helper(s)
class Solution(object):
	def numDecodings(self, s): # 稍加改进，使用DP就可以很快了 87.80%
		"""
		:type s: str
		:rtype: int
		"""
		if not s or s[0] == '0': return 0
		illegal_strs = ['00', '30', '40', '50', '60', '70', '80', '90']
		for illegal_str in illegal_strs:
			if illegal_str in s: return 0

		def helper(s):
			l = len(s)
			if len(s) < 2: return 1
			res = [1 for i in range(l + 1)]
			for i in range(2, l + 1):
				if int(s[i-2:i]) <= 26: res[i] = res[i - 2] + res[i - 1]
				else: res[i] = res[i - 1]
			print(res)
			return res[-1]

		if '0' in s:
			s_ = s.split('0')
			res = 1
			for _ in s_:
				res *= helper(_[:-1])
			return res

		return helper(s)
class Solution(object):
	def numDecodings(self, s): # 别人的代码，他妈的，为什么
		v, w, p = 0, int(s>''), ''
		for d in s:
			v, w, p = w, (d>'0')*w + (9<int(p+d)<27)*v, d
		return w

#————————————————19/4/14———————————————————
# 746. Min Cost Climbing Stairs
class Solution(object): # 37.81% 不知道为什么这么慢
	def minCostClimbingStairs(self, cost):
		"""
		:type cost: List[int]
		:rtype: int
		"""
		p = c = 0
		for _ in cost:
			c, p = min(_ + c, _ + p), c
		return min(c, p)

# 96. Unique Binary Search Trees
class Solution(object):
	def numTrees(self, n): # 66.69%
		"""
		:type n: int
		:rtype: int
		"""
		if n < 2: return n
		r = [0] * (n + 1)
		r[0] = r[1] = 1
		for i in range(2, n + 1):
			for j in range(i):
				r[i] += r[j] * r[i - j - 1]
		return r[-1]

#————————————————19/4/15———————————————————
# 101. Symmetric Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 88.86%
	def isSymmetric(self, root):
		"""
		:type root: TreeNode
		:rtype: bool
		"""
		def helper(l, r):
			if not l or not r: return l == r
			if l.val != r.val: return False
			return helper(l.left, r.right) and helper(l.right, r.left)
		return root is None or helper(root.left, root.right)

# 112. Path Sum
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 78.42%
	def hasPathSum(self, root, sum):
		"""
		:type root: TreeNode
		:type sum: int
		:rtype: bool
		"""
		if not root: return False
		if not root.left and not root.right: return sum == root.val
		return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

# 39. Combination Sum 明天再做
class Solution(object):
	def combinationSum(self, candidates, target):
		"""
		:type candidates: List[int]
		:type target: int
		:rtype: List[List[int]]
		"""
		res = []

#————————————————19/4/16———————————————————
# 39. Combination Sum 今天继续做
class Solution(object):
	def combinationSum(self, candidates, target): # 98.98% 抄别人的🐎的
		"""
		:type candidates: List[int]
		:type target: int
		:rtype: List[List[int]]
		"""
		res = []
		def helper(c, s, t, r):
			if not t:
				res.append(r)
				return
			for i in range(s, len(c)):
				if c[i] > t: break
				helper(c, i, t - c[i], r + [c[i]])

		helper(sorted(candidates), 0, target, [])
		return res

# 235. Lowest Common Ancestor of a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object):
	def lowestCommonAncestor(self, root, p, q): # 72.14%
		"""
		:type root: TreeNode
		:type p: TreeNode
		:type q: TreeNode
		:rtype: TreeNode
		"""
		if root in [p, q]: return root
		if root.val < max(p.val, q.val) and root.val > min(p.val, q.val):
			return root
		return self.lowestCommonAncestor(root.left if root.val > p.val else root.right, p, q)

#————————————————19/4/17———————————————————
# 236. Lowest Common Ancestor of a Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 99.54%
	def lowestCommonAncestor(self, root, p, q):
		"""
		:type root: TreeNode
		:type p: TreeNode
		:type q: TreeNode
		:rtype: TreeNode
		"""
		def helper(a, b, c = None):
			if not a: return False
			if a == b: return True
			if c and a == c: return True
			return helper(a.left, b, c) or helper(a.right, b, c)
		if helper(p, q): return p
		if helper(q, p): return q
		stack = []
		stack.append(root)
		while stack:
			cur = stack.pop()
			if helper(cur.left, p, q) and helper(cur.right, p, q): return cur
			if cur.left: stack.append(cur.left)
			if cur.right: stack.append(cur.right)

class Solution(object): # 47.81%
	ans = None
	def lowestCommonAncestor(self, root, p, q):
		"""
		:type root: TreeNode
		:type p: TreeNode
		:type q: TreeNode
		:rtype: TreeNode
		"""
		def helper(cur):
			if not cur: return False
			l = helper(cur.left)
			r = helper(cur.right)
			m = cur in [p, q]
			if l + m + r >= 2: self.ans = cur
			return l or r or m
		helper(root)
		return self.ans

#————————————————19/4/18———————————————————
# 771. Jewels and Stones
class Solution(object): # 99.81%
	def numJewelsInStones(self, J, S):
		"""
		:type J: str
		:type S: str
		:rtype: int
		"""
		res = 0
		for ch in S:
			if ch in J:
				res += 1
		return res

# 709. To Lower Case
class Solution(object): # 11.48%
	def toLowerCase(self, str):
		"""
		:type str: str
		:rtype: str
		"""
		ans = ''
		for ch in str:
			ans += chr(ord(ch) + 32) if 'Z' >= ch >= 'A' else ch
		return ans

# 804. Unique Morse Code Words
class Solution(object): # 67.44%
	def uniqueMorseRepresentations(self, words):
		"""
		:type words: List[str]
		:rtype: int
		"""
		d = [".-","-...","-.-.","-..",".","..-.","--.","....","..",\
			".---","-.-",".-..","--","-.","---",".--.","--.-",".-.",\
			"...","-","..-","...-",".--","-..-","-.--","--.."]
		def helper(word):
			res = ''
			for ch in word:
				res += d[ord(ch) - ord('a')]
			return res
		ans = []
		for word in words:
			res = helper(word)
			if res not in ans:
				ans.append(res)

		return len(ans)

#————————————————19/4/19———————————————————
# 34. Find First and Last Position of Element in Sorted Array
class Solution(object): # 38.61 %
	def searchRange(self, nums, target):
		"""
		:type nums: List[int]
		:type target: int
		:rtype: List[int]
		"""
		if not nums: return [-1, -1]
		l, r = 0, len(nums) - 1
		while l <= r:
			m = (l + r) // 2
			if nums[m] > target:
				r = m - 1
			elif nums[m] < target:
				l = m + 1
			else:
				lr = rl = m
				while l < lr:
					mm = (l + lr) // 2
					if nums[mm] < target:
						l = mm + 1
					else:
						lr = mm
				while r > rl:
					mm = (r + rl + 1) // 2
					if nums[mm] > target:
						r = mm - 1
					else:
						rl = mm
				return [l, r]

		return [-1, -1]
class Solution(object): # 32.69%
	def searchRange(self, nums, target):
		"""
		:type nums: List[int]
		:type target: int
		:rtype: List[int]
		"""
		l = r = -1
		lf = 0
		for i in range(len(nums)):
			if nums[i] > target: break
			if nums[i] == target:
				if not lf:
					l, lf = i, 1
				r = i
		return [l, r]

# 40. Combination Sum II
class Solution(object): # 99.62%
	def combinationSum2(self, candidates, target):
		"""
		:type candidates: List[int]
		:type target: int
		:rtype: List[List[int]]
		"""
		ans = []
		def helper(c, s, t, r):
			if not t:
				ans.append(r)
				return
			for i in range(s, len(c)):
				if i > s and c[i] == c[i - 1]: continue
				if c[i] > t: break
				helper(c, i + 1, t - c[i], r + [c[i]])
		helper(sorted(candidates), 0, target, [])
		return ans

#————————————————19/4/20———————————————————
# 595. Big Countries (it is a MySQL problem)
# Write your MySQL query statement below
SELECT name, population, area # 97.25%
FROM World
WHERE area > 3000000 OR population > 25000000

# 929. Unique Email Addresses
class Solution(object): # 17.99%
	def numUniqueEmails(self, emails):
		"""
		:type emails: List[str]
		:rtype: int
		"""
		res = set()
		for email in emails:
			local_real = ''
			local, domin = email.split('@')
			for ch in local:
				if ch == '+': break
				if ch != '.': local_real += ch
			email_real = local_real + '@' + domin
			res.add(email_real)
		return len(res)
class Solution(object): # 53.03%
	def numUniqueEmails(self, emails):
		"""
		:type emails: List[str]
		:rtype: int
		"""
		def parse(email):
			local, domin = email.split('@')
			local = local.split('+')[0].replace('.', '')
			return "{}@{}".format(local, domin)

		return len(set(map(parse, emails)))

#————————————————19/4/21———————————————————
# 905. Sort Array By Parity
class Solution(object): # 49.45%
	def sortArrayByParity(self, A):
		"""
		:type A: List[int]
		:rtype: List[int]
		"""
		even, odd = [], []
		for n in A:
			if n % 2: odd.append(n)
			else: even.append(n)
		return even + odd

# 977. Squares of a Sorted Array
class Solution(object): # 38.18%
	def sortedSquares(self, A):
		"""
		:type A: List[int]
		:rtype: List[int]
		"""
		return sorted([x ** 2 for x in A])

#————————————————19/4/22———————————————————
# 807. Max Increase to Keep City Skyline
class Solution(object): # 41.68%
	def maxIncreaseKeepingSkyline(self, grid):
		"""
		:type grid: List[List[int]]
		:rtype: int
		"""
		ans, h, w = 0, len(grid), len(grid[0])
		col_sky = [max(col) for col in grid]
		row_sky = [max([grid[i][j] for i in range(h)]) for j in range(w)]

		for i in range(h):
			for j in range(w):
				ans += min(row_sky[i], col_sky[j]) - grid[i][j]

		return ans

# 938. Range Sum of BST
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 80.54%
	def rangeSumBST(self, root, L, R):
		"""
		:type root: TreeNode
		:type L: int
		:type R: int
		:rtype: int
		"""
		if not root: return 0
		ans, stack = 0, []
		stack.append(root)
		while stack:
			node = stack.pop()
			val = node.val
			if R >= val >= L:
				ans += val
				if node.left: stack.append(node.left)
				if node.right: stack.append(node.right)
			else:
				if val > R and node.left: stack.append(node.left)
				if val < L and node.right: stack.append(node.right)
		return ans
class Solution(object): # 80.54%
	def rangeSumBST(self, root, L, R):
		"""
		:type root: TreeNode
		:type L: int
		:type R: int
		:rtype: int
		"""
		if not root: return 0
		if root.val > R: return self.rangeSumBST(root.left, L, R)
		elif root.val < L: return self.rangeSumBST(root.right, L, R)
		return root.val + self.rangeSumBST(root.left, L, R) + self.rangeSumBST(root.right, L, R)

#————————————————19/4/23———————————————————
# 535. Encode and Decode TinyURL
import random, string
class Codec: # 62.42%
	def __init__(self):
		self.urls = dict()

	def encode(self, longUrl):
		"""Encodes a URL to a shortened URL.

		:type longUrl: str
		:rtype: str
		"""
		key = ''.join(random.sample(string.ascii_letters + string.digits, 6))
		if key in self.urls:
			while key in self.urls:
				key = ''.join(random.sample(string.ascii_letters + string.digits, 6))
		self.urls[key] = longUrl
		return 'http://tinyurl.com/' + key


	def decode(self, shortUrl):
		"""Decodes a shortened URL to its original URL.

		:type shortUrl: str
		:rtype: str
		"""
		return self.urls[shortUrl.split('/')[-1]]

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

# 654. Maximum Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 42.84 %
	def constructMaximumBinaryTree(self, nums):
		"""
		:type nums: List[int]
		:rtype: TreeNode
		"""
		if not nums: return None
		m = max(nums)
		m_p = nums.index(m)
		root = TreeNode(m)
		root.left = self.constructMaximumBinaryTree(nums[:m_p])
		root.right = self.constructMaximumBinaryTree(nums[m_p + 1:])
		return root

#————————————————19/4/24———————————————————
# 701. Insert into a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object): # 70.17%
	def insertIntoBST(self, root, val):
		"""
		:type root: TreeNode
		:type val: int
		:rtype: TreeNode
		"""
		if not root: return TreeNode(val)
		if root.val > val: root.left = self.insertIntoBST(root.left, val)
		else: root.right = self.insertIntoBST(root.right, val)
		return root


# 55. Jump Game
class Solution(object): # 25.89%
	def canJump(self, nums):
		"""
		:type nums: List[int]
		:rtype: bool
		"""
		if len(nums) < 2: return True
		c, n, l = 0, 1, len(nums) - 1
		while nums[c]:
			if nums[c] >= l - c: return True
			n = c + 1
			for i in range(2, nums[c] + 1):
				if nums[c + i] >= nums[n] or nums[n] - nums[c + i] <= c + i - n:
					n = c + i
			c = n

		return False
class Solution(object): # 41.80% 牛皮撒
	def canJump(self, nums):
		"""
		:type nums: List[int]
		:rtype: bool
		"""
		m = 0
		for i, n in enumerate(nums):
			if i > m: return False
			m = max(m, i + n)
		return True
class Solution(object): # 46.21%
	def canJump(self, nums):
		"""
		:type nums: List[int]
		:rtype: bool
		"""
		goal = len(nums) - 1
		for i in range(len(nums) - 1)[::-1]:
			if nums[i] + i >= goal:
				goal = i
		return not goal

#————————————————19/4/25———————————————————
# 48. Rotate Image
class Solution(object): # 73.38%
	def rotate(self, matrix):
		"""
		:type matrix: List[List[int]]
		:rtype: None Do not return anything, modify matrix in-place instead.
		"""
		# matrix[:] = zip(*matrix[::-1])
		matrix[:] = [[row[i] for row in matrix[::-1]] for i in range(len(matrix))]
class Solution(object): # 73.38%
	def rotate(self, matrix):
		"""
		:type matrix: List[List[int]]
		:rtype: None Do not return anything, modify matrix in-place instead.
		"""
		n = len(matrix)
		for i in range(n // 2):
			matrix[i], matrix[~i] = matrix[~i], matrix[i]
		for i in range(n):
			for j in range(i + 1, n):
				matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

# 43. Multiply Strings
class Solution(object): # 30.07% 作弊了
	def multiply(self, num1, num2):
		"""
		:type num1: str
		:type num2: str
		:rtype: str
		"""
		ans = 0
		for i1, n1 in enumerate(num1[::-1]):
			res = 0
			for i2, n2 in enumerate(num2[::-1]):
				res += (ord(n2) - ord('0')) * (ord(n1) - ord('0')) * (10 ** i2)
			ans += res * (10 ** i1)
		return ans
class Solution(object): # 52.87%
	def multiply(self, num1, num2):
		"""
		:type num1: str
		:type num2: str
		:rtype: str
		"""
		l1, l2 = len(num1), len(num2)
		res = [0 for i in range(l1 + l2)]
		for i in range(l1)[::-1]:
			for j in range(l2)[::-1]:
				m, p1, p2 = (ord(num2[j]) - ord('0')) * (ord(num1[i]) - ord('0')), i + j, i + j + 1
				s = m + res[p2]
				res[p1], res[p2] = res[p1] + s // 10, s % 10
		while not res[0] and res != [0]: res.remove(0)
		return ''.join(map(str, res))

#————————————————19/4/26———————————————————
# 47. Permutations II
class Solution(object): # 8.78%
	def permuteUnique(self, nums):
		"""
		:type nums: List[int]
		:rtype: List[List[int]]
		"""
		ans = []
		def helper(nums, _):
			if not nums:
				if _ not in ans:
					ans.append(_)
				return
			for i in range(l):
				helper(nums[:i] + nums[i+1:], i, l, _ + [nums[i]])

		helper(nums, 0, [])
		return ans
class Solution(object): # 15.91%
	def permuteUnique(self, nums):
		"""
		:type nums: List[int]
		:rtype: List[List[int]]
		"""
		def helper(n):
			if not n: return [[]]
			return [_ + [n[i]] for i in range(len(n)) for _ in helper(n[:i] + n[i+1:])]
		return [list(_) for _ in set([tuple(_) for _ in helper(nums)])]
class Solution(object): # 100.00% niubility
	def permuteUnique(self, nums):
		"""
		:type nums: List[int]
		:rtype: List[List[int]]
		"""
		perms = [[]]
		for i, n in enumerate(nums):
			perms = [p[:i] + [n] + p[i:] for p in perms for i in range((p + [n]).index(n) + 1)]
		return [list(p) for p in set([tuple(p) for p in perms])]

# 49. Group Anagrams
class Solution(object): # 91.24%
	def groupAnagrams(self, strs):
		"""
		:type strs: List[str]
		:rtype: List[List[str]]
		"""
		s, r = [''.join(sorted([ch for ch in str])) for str in strs], {}
		for i, s_ in enumerate(s):
			if s_ not in r:
				r[s_] = [i]
			else:
				r[s_].append(i)

		return [[strs[i] for i in v] for v in r.values()]

#————————————————19/4/27———————————————————
# 50. Pow(x, n)
class Solution: # 99.74%
	def myPow(self, x: float, n: int) -> float:
		p, (f, n) = 1, (1, n) if n > 0 else (0, -n)
		while n:
			if n & 1: p *= x
			x *= x
			n >>= 1
		return p if f else 1 / p

# 54. Spiral Matrix
class Solution: # 77.19%
	def spiralOrder(self, m: List[List[int]]) -> List[int]:
		r = []
		while m:
			r += m.pop(0)
			m = [*zip(*m)][::-1]
		return r
class Solution: # 77.19% 大神写的一行AC代码
	def spiralOrder(self, m: List[List[int]]) -> List[int]:
		return m and [*m.pop(0)] + self.spiralOrder([*zip(*m)][::-1])

#————————————————19/4/28———————————————————
# 1033. Moving Stones Until Consecutive
class Solution(object):
	def numMovesStones(self, a, b, c):
		"""
		:type a: int
		:type b: int
		:type c: int
		:rtype: List[int]
		"""
		a, b, c = sorted([a, b, c])
		return [2 if c - b > 2 and b - a > 2 else int(c - b > 1 or b - a > 1), c - a - 2]

# 1034. Coloring A Border
class Solution(object):
	def colorBorder(self, grid, r0, c0, color):
		"""
		:type grid: List[List[int]]
		:type r0: int
		:type c0: int
		:type color: int
		:rtype: List[List[int]]
		"""
		h, w, c = len(grid), len(grid[0]), grid[r0][c0]
		m = [[0 for i in range(w)] for j in range(h)]
		def helper(g, x, y, c, m, h, w):
			if x < 0 or x >= h or y < 0 or y >= w or m[x][y]: return
			if g[x][y] != c:
				m[x][y] = -1
				return
			m[x][y] = 1
			helper(g, x + 1, y, c, m, h, w)
			helper(g, x - 1, y, c, m, h, w)
			helper(g, x, y + 1, c, m, h, w)
			helper(g, x, y - 1, c, m, h, w)
		helper(grid, r0, c0, c, m, h, w)
		for i in range(h):
			for j in range(w):
				if m[i][j] == 1 and not (h - 1 > i > 0 and w - 1 > j > 0 and m[i - 1][j] == 1 and m[i + 1][j] == 1 and m[i][j - 1] == 1 and m[i][j + 1] == 1):
					grid[i][j] = color
		return grid

# 500. Keyboard Row
class Solution(object):
	def findWords(self, words):
		"""
		:type words: List[str]
		:rtype: List[str]
		"""
		def helper(ch):
			if ch in 'qwertyuiopQWERTYUIOP': return 1
			elif ch in 'asdfghjklASDFGHJKL': return 2
			return 3
		res = []
		for word in words:
			l, f = helper(word[0]), 1
			for ch in word:
				if helper(ch) != l:
					f = 0
					break
			if f: res.append(word)
		return res

# 508. Most Frequent Subtree Sum
# Definition for a binary tree node.
# class TreeNode(object):
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution(object):
	def findFrequentTreeSum(self, root):
		"""
		:type root: TreeNode
		:rtype: List[int]
		"""
		if not root: return []
		res, ans = [], {}
		def helper(root):
			if not root: return 0
			s = root.val + helper(root.left) + helper(root.right)
			res.append(s)
			return s
		helper(root)
		for _ in res:
			if _ not in ans: ans[_] = 1
			else: ans[_] += 1
		m, r = max(ans.values()), []
		for _ in ans.keys():
			if ans[_] == m: r.append(_)
		return r

#————————————————19/4/29———————————————————
# 59. Spiral Matrix II
class Solution(object): # 100.00%
	def generateMatrix(self, n):
		"""
		:type n: int
		:rtype: List[List[int]]
		"""
		steps, way, res = [], [[0, 1], [1, 0], [0, -1], [-1, 0]], [[0 for i in range(n)] for j in range(n)]
		for i in range(1, n): steps += [i, i]
		steps.append(n)
		x, y, w, c = 0, -1, 0, 0
		while steps:
			step = steps.pop()
			for i in range(step):
				c, x, y = c + 1, x + way[w][0], y + way[w][1]
				res[x][y] = c
			w = (w + 1) % 4
		return res
class Solution(object): # 100.00% niubility
	def generateMatrix(self, n):
		A, lo = [], n*n+1
		while lo > 1:
			lo, hi = lo - len(A), lo
			A = [range(lo, hi)] + zip(*A[::-1])
		return A
# 961. N-Repeated Element in Size 2N Array
class Solution(object): # 33.33%
	def repeatedNTimes(self, A):
		"""
		:type A: List[int]
		:rtype: int
		"""
		r = set()
		for n in A:
			if n in r: return n
			r.add(n)

#————————————————19/4/30———————————————————
# 56. Merge Intervals
class Solution: # 69.89%
	def merge(self, intervals: List[List[int]]) -> List[List[int]]:
		intervals.sort()
		i = 0
		while i < len(intervals) - 1:
			if intervals[i][1] >= intervals[i + 1][0]:
				intervals[i] = [intervals[i][0], max(intervals[i + 1][1], intervals[i][1])]
				intervals.pop(i + 1)
			else: i += 1
		return intervals
class Solution: # 100.00%
	def merge(self, intervals):
		out = []
		for i in sorted(intervals, key=lambda i: i[0]):
			if out and i[0] <= out[-1][1]:
				out[-1][1] = max(out[-1][1], i[1])
			else:
				out += i,
		return out

# 60. Permutation Sequence
class Solution(object): # 95.06%
	def getPermutation(self, n, k):
		def helper(r, k):
			if k == 1: return r
			f = reduce(lambda x,y: x*y, [i for i in range(1, len(r))])
			a, b = (k-1)//f, k % f if k % f else f
			return r[a] + helper(r[:a] + r[a+1:], b)
		return helper(''.join([str(i) for i in range(1, n + 1)]), k)

#————————————————19/5/1———————————————————
# 832. Flipping an Image
class Solution: # 77.63%
	def flipAndInvertImage(self, A: List[List[int]]) -> List[List[int]]:
		return [list(map(lambda x: 0 if x else 1, row[::-1])) for row in A]

# 1008. Construct Binary Search Tree from Preorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution: # 100.00%
	def bstFromPreorder(self, preorder: List[int]) -> TreeNode:
		def helper(r, n):
			if not r: return TreeNode(n)
			if r.val > n: r.left = helper(r.left, n)
			else: r.right = helper(r.right, n)
			return r
		r = None
		for n in preorder:
			r = helper(r, n)
		return r

#————————————————19/5/2———————————————————
# 61. Rotate List
# Definition for singly-linked list.
# class ListNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.next = None

class Solution: # 86.05%
	def rotateRight(self, head: ListNode, k: int) -> ListNode:
		if not head: return None
		l, p = 0, head
		while p:
			l, p = l + 1, p.next
		k, p = l - k % l if k and k % l else 0, head
		if not k or l < 2: return head
		while k:
			p, k = p.next, k - 1
		q = p
		while q.next:
			q = q.next
		q.next = head
		while head.next != p:
			head = head.next
		head.next = None
		return p

# 63. Unique Paths II
class Solution: # Time Limit Exceeded
	def uniquePathsWithObstacles(self, A: List[List[int]]) -> int:
		if A[0][0] or not A: return 0
		self.r = 0
		def dfs(x, y, h, w, A):
			if x == h and y == w:
				self.r += 1
				return
			if x + 1 <= h and not A[x + 1][y]: dfs(x + 1, y, h, w, A)
			if y + 1 <= w and not A[x][y + 1]: dfs(x, y + 1, h, w, A)
		dfs(0, 0, len(A) - 1, len(A[0]) - 1, A)
		return self.r
class Solution: # 76.53%
	def uniquePathsWithObstacles(self, A: List[List[int]]) -> int:
		if A[0][0] or A[-1][-1] or not A: return 0
		h, w = len(A), len(A[0])
		m = [[1 for j in range(w)] for i in range(h)]
		for i in range(h):
			for j in range(w):
				if i == 0 and j == 0: m[i][j] == 1
				else: m[i][j] = 0 if A[i][j] else (m[i - 1][j] if i > 0 else 0) + (m[i][j - 1] if j > 0 else 0)

		return m[-1][-1]

#————————————————19/5/3———————————————————
# 71. Simplify Path
class Solution: # 97.84%
	def simplifyPath(self, path: str) -> str:
		while '//' in path: path = path.replace('//', '/')
		file, res = path.split('/'), []
		while '' in file: file.remove('')
		while '.' in file: file.remove('.')
		for f in file:
			if f != '..': res.append(f)
			elif res: res.pop()
		return '/' + '/'.join(res)

# 73. Set Matrix Zeroes
class Solution: # 82.83%
	def setZeroes(self, A: List[List[int]]) -> None:
		"""
		Do not return anything, modify matrix in-place instead.
		"""
		h, w = len(A), len(A[0])
		row, col = [], []
		for i in range(h):
			for j in range(w):
				if not A[i][j]:
					if i not in row: row.append(i)
					if j not in col: col.append(j)

		for r in row:
			for j in range(w):
				A[r][j] = 0

		for c in col:
			for i in range(h):
				A[i][c] = 0

# 77. Combinations
class Solution:
	def combine(self, n: int, k: int) -> List[List[int]]:
		self.res = []
		def helper(r, nums, k):
			if k == 0:
				self.res.append(r)
				return
			for i in range(len(nums)):
				helper(r + [nums[i]], nums[i+1:], k - 1)

		helper([], [i + 1 for i in range(n)], k)
		return self.res
class Solution: # 50.15%
	def combine(self, n: int, k: int) -> List[List[int]]:
		self.res = []
		def helper(r, s, n, k):
			if k == 0:
				self.res.append(r)
				return
			for i in range(s, n + 1):
				helper(r + [i], i + 1, n, k - 1)

		helper([], 1, n, k)
		return self.res
class Solution: # 69.27%
	def combine(self, n: int, k: int) -> List[List[int]]:
		if k == 0:
			return [[]]
		return [pre + [i] for i in range(k, n+1) for pre in self.combine(i-1, k-1)]
class Solution: # 11.05%
	def combine(self, n, k):
		combs = [[]]
		for _ in range(k):
			combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
		return combs

#————————————————19/5/4———————————————————
# 45. Jump Game II
class Solution: # 66.75%
	def jump(self, nums: List[int]) -> int:
		c, r, l = 0, 0, len(nums) - 1
		while c != l:
			if c + nums[c] >= l: c = l
			else:
				m = c + nums[c]
				for i in range(c + 1, c + nums[c] + 1):
					if m <= i + nums[i]:
						m, c = i + nums[i], i
			r += 1
		return r
class Solution: # 51.41%
	def jump(self, nums: List[int]) -> int:
		c, n, r, l = 0, 0, 0, len(nums)
		while n < l - 1:
			c, n = n, max(i + nums[i] for i in range(c, n + 1))
			r += 1
		return r

# 74. Search a 2D Matrix
class Solution: # 82.23%
	def searchMatrix(self, A, t):
		if not A or not A[0]: return False
		if t < A[0][0] or t > A[-1][-1]: return False
		h, w, x, y = len(A), len(A[0]), -1, -1
		u, d, l, r = 0, h - 1, 0, w - 1
		while u <= d:
			m = (u + d) // 2
			if A[m][0] <= t <= A[m][-1]:
				x = m
				break
			if A[m][0] > t: d = m - 1
			elif A[m][-1] < t: u = m + 1
		if x == -1: return False
		while l <= r:
			m = (l + r) // 2
			if A[x][m] == t:
				y = m
				break
			elif A[x][m] < t: l = m + 1
			else: r = m - 1
		if y == -1: return False
		return True

#————————————————19/5/5———————————————————
# 78. Subsets
class Solution: # 87.20%
	def subsets(self, nums):
		r = [[]]
		for n in nums:
			r += [_ + [n] for _ in r]
		return r

# 79. Word Search
class Solution: # 56.28%
	def exist(self, b, w):
		x, y = len(b), len(b[0])
		for i in range(x):
			for j in range(y):
				if self.dfs(b, i, j, w):
					return True
		return False

	def dfs(self, b, x, y, w):
		if not w: return True
		if x < 0 or x >= len(b) or y < 0 or y >= len(b[0]) or b[x][y] != w[0]: return False
		t, b[x][y] = b[x][y], '#'
		res = self.dfs(b, x + 1, y, w[1:]) or self.dfs(b, x - 1, y, w[1:]) or self.dfs(b, x, y + 1, w[1:]) or self.dfs(b, x, y - 1, w[1:])
		b[x][y] = t
		return res

# Weekly Contest 135
# 1037. Valid Boomerang
class Solution:
	def isBoomerang(self, points: List[List[int]]) -> bool:
		A, B, C = points
		if A == B or B == C or A == C or \
			(C[1] - B[1]) * (B[0] - A[0]) == (B[1] - A[1]) * (C[0] - B[0]): return False
		return True

# 1038. Binary Search Tree to Greater Sum Tree
# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution:
	val = 0
	def bstToGst(self, r: TreeNode) -> TreeNode:
		if r.right: self.bstToGst(r.right)
		r.val = self.val = self.val + r.val
		if r.left: self.bstToGst(r.left)
		return r

#————————————————19/5/6———————————————————
# 80. Remove Duplicates from Sorted Array II
class Solution: # 95.73%
	def removeDuplicates(self, nums):
		r, t, p, c, l = 0, 0, 0, 0, len(nums)
		while t < l:
			c = c + 1 if nums[t] == nums[p] else 1
			if c > 2:
				t += 1
				continue
			if t > r:
				nums[r] = nums[t]
			p, r, t = r, r+1, t+1

		return r
class Solution: # 95.73% niubility
	def removeDuplicates(self, nums):
		r = 0
		for n in nums:
			if r < 2 or n > nums[r-2]:
				nums[r] = n
				r += 1
		return r

# 86. Partition List
# Definition for singly-linked list.
# class ListNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.next = None

class Solution: # 97.54%
	def partition(self, h, x):
		h1 = l1 = ListNode(0)
		h2 = l2 = ListNode(0)
		while h:
			if h.val < x:
				l1.next = h
				l1 = h
			else:
				l2.next = h
				l2 = h
			h = h.next
		l2.next = None
		l1.next = h2.next
		return h1.next

# 89. Gray Code
class Solution: # 5.18%
	def grayCode(self, n):
		r = [0]
		for i in range(2 ** n - 1):
			for i in range(n):
				if r[-1]^(2 ** i) not in r:
					r.append(r[-1]^(2 ** i))
					break
		return r
class Solution: # 82.25%
	def grayCode(self, n):
		r = []
		for i in range(1<<n):
			r.append(i^(i>>1))
		return r
class Solution: # 34.71%
	def grayCode(self, n):
		return self.grayCode(n-1) + [2**(n-1) + _ for _ in self.grayCode(n-1)[::-1]] if n else [0]

#————————————————19/5/7———————————————————
# 93. Restore IP Addresses
class Solution: # 100.00%
	def restoreIpAddresses(self, s):
		l, a = len(s), []
		if l > 12 or l < 4: return a
		def helper(r, s, k):
			if not k:
				if not s:
					a.append(r[1:])
				return
			if s:
				helper(r + '.' + s[0], s[1:], k - 1)
			if len(s) > 1 and s[0] != '0':
				helper(r + '.' + s[:2], s[2:], k - 1)
			if len(s) > 2 and s[0] != '0' and int(s[:3]) < 256:
				helper(r + '.' + s[:3], s[3:], k - 1)
		helper('', s, 4)
		return a

# 94. Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution: # 100.00%
	def inorderTraversal(self, root):
		res, stack = [], []
		while True:
			while root:
				stack.append(root)
				root = root.left
			if not stack:
				return res
			node = stack.pop()
			res.append(node.val)
			root = node.right

# 868. Binary Gap
class Solution: # 99.42%
	def binaryGap(self, N):
		r = bin(N)[:2].split('1')[1:-1]
		if not r: return 0
		return max(map(len, r))+1

# 869. Reordered Power of 2
class Solution: # 100.00%
	r = ['1', '2', '4', '8', '16', '23', '46', '128', '256', '125', '0124', '0248', '0469', '1289', '13468', '23678', '35566', '011237', '122446', '224588', '0145678', '0122579', '0134449', '0368888', '11266777', '23334455', '01466788', '112234778', '234455668', '012356789']
	def reorderedPowerOf2(self, N):
		return ''.join(sorted([_ for _ in str(N)])) in self.r

# 870. Advantage Shuffle
class Solution: # 87.43%
	def advantageCount(self, A, B):
		r = [0 for _ in A]
		A.sort()
		for b, i in sorted((b, i) for i, b in enumerate(B))[::-1]:
			if A[-1] > b:
				r[i] = A.pop()
			else:
				r[i] = A.pop(0)
		return r

#————————————————19/5/8———————————————————
# 871. Minimum Number of Refueling Stops
class Solution: # 12.77%
	def minRefuelStops(self, target, startFuel, stations):
		dp = [startFuel] + [0] * len(stations)
		for i in range(len(stations)):
			for t in range(i + 1)[::-1]:
				if dp[t] >= stations[i][0]:
					dp[t + 1] = max(dp[t + 1], dp[t] + stations[i][1])
		for t, d in enumerate(dp):
			if d >= target:
				return t
		return -1

# 171. Excel Sheet Column Number
class Solution: # 100.00%
	def titleToNumber(self, s):
		r = 0
		for i in range(len(s)):
			print(r)
			r += (ord(s[~i]) - ord('A') + 1) * (26 ** i)
		return r

#————————————————19/5/10———————————————————
# 102. Binary Tree Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution: # 97.31%
	def levelOrder(self, root):
		queue, res = [], []
		def helper(root, level):
			if not root: return
			if len(res) <= level:
				res.append([])
				res[level].append(root.val)
			else: res[level].append(root.val)
			helper(root.left, level+1)
			helper(root.right, level+1)
		helper(root, 0)
		return res
class Solution: # 97.31%
	def levelOrder(self, root):
		res, level = [], [root]
		while root and level:
			res.append([node.val for node in level])
			level = [kid for node in level for kid in (node.left, node.right) if kid]
		return res

# 103. Binary Tree Zigzag Level Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#	 def __init__(self, x):
#		 self.val = x
#		 self.left = None
#		 self.right = None

class Solution: # 99.24%
	def zigzagLevelOrder(self, root):
		res, level, f = [], [root], 0
		while root and level:
			res.append([node.val for node in (level[::-1] if f else level)])
			level, f = [kid for node in level for kid in (node.left, node.right) if kid], 0 if f else 1
		return res

# 470. Implement Rand10() Using Rand7()
# The rand7() API is already defined for you.
# def rand7():
# @return a random integer in the range 1 to 7

class Solution: # 14.44%
	def rand10(self):
		"""
		:rtype: int
		"""
		i, j = 0, 7
		while i < 6:
			i = rand7()
		while j > 5:
			j = rand7()
		return j if (i&1) else j+5

#————————————————19/5/11———————————————————
# 116. Populating Next Right Pointers in Each Node
"""
# Definition for a Node.
class Node:
	def __init__(self, val, left, right, next):
		self.val = val
		self.left = left
		self.right = right
		self.next = next
"""
class Solution: # 95.70%
	def connect(self, root: 'Node') -> 'Node':
		level = [root]
		while root and level:
			for i in range(len(level)-1):
				level[i].next = level[i+1]
			level = [kid for node in level for kid in (node.left, node.right) if kid]
		return root

# 119. Pascal's Triangle II
class Solution: # 85.27%
	def getRow(self, rowIndex: int) -> List[int]:
		res = [1]
		for i in range(1, rowIndex+1):
			res.append(1)
			for j in range(1, i)[::-1]:
				res[j] = res[j] + res[j-1]
		return res

#————————————————19/5/12———————————————————
# 172. Factorial Trailing Zeroes
class Solution: # 97.76%
	def trailingZeroes(self, n: int) -> int:
		r = 0
		while n:
			r += n // 5
			n //= 5
		return r

# 202. Happy Number
class Solution: # 98.04%
	def isHappy(self, n: int) -> bool:
		res = []
		while n not in res:
			res.append(n)
			n = sum([int(c)**2 for c in str(n)])
		return 1 in res

#————————————————19/5/13———————————————————
# 135. Candy
class Solution: # 86.87%
	def candy(self, ratings: List[int]) -> int:
		res = [1 for _ in ratings]
		for i in range(1, len(ratings)):
			if ratings[i] > ratings[i-1]:
				res[i] = res[i-1] + 1
		for i in range(1, len(ratings))[::-1]:
			if ratings[i-1] > ratings[i]:
				res[i-1] = max(res[i-1], res[i]+1)
		return sum(res)

# 392. is Subsequence
class Solution: # 91.47%
	def isSubsequence(self, s: str, t: str) -> bool:
		if not s: return True
		if s[0] not in t: return False
		return self.isSubsequence(s[1:], t[t.index(s[0])+1:])

#————————————————19/5/15———————————————————
# 179. Largest Number
class Solution: # 17.10%
	def largestNumber(self, nums) -> str:
		nums = [str(_) for _ in nums]
		for i in range(len(nums)):
			for j in range(len(nums)-i-1):
				if nums[j] + nums[j+1] < nums[j+1] + nums[j]:
					nums[j], nums[j+1] = nums[j+1], nums[j]
		r = ''.join(nums)
		return r if r[0] != '0' else '0'
class Solution: # 99.31% python2
	def largestNumber(self, nums):
		r = ''.join(sorted(map(str, nums), lambda x, y: [1, -1][x+y > y+x]))
		return r if r[0] != '0' else '0'

# 183. Customers Who Never Order
# Write your MySQL query statement below
SELECT C.Name Customers FROM # 27.81%
Customers C
Where C.Id NOT IN
(SELECT O.CustomerId FROM
Orders O)
# Write your MySQL query statement below
SELECT C.Name Customers # 89.86%
FROM Customers C
LEFT JOIN Orders O
ON C.Id = O.CustomerId
WHERE O.CustomerId IS NULL

#————————————————19/5/16———————————————————
# 160. Intersection of Two Linked Lists
# Definition for singly-linked list.
# class ListNode(object):
#	def __init__(self, x):
#		self.val = x
#		self.next = None

class Solution(object): # TLE
	def getIntersectionNode(self, headA, headB):
		"""
		:type head1, head1: ListNode
		:rtype: ListNode
		"""
		if not headA or not headB: return None
		pa, pb = headA, headB
		while pa is not pb:
			pa = headA if pa is None else pa.next
			pb = headB if pb is None else pb.next
		return pa
class Solution(object): # 90.82% Brilliant Function
	def getIntersectionNode(self, headA, headB):
		pa, pb = headA, headB
		la, lb = 0, 0
		while pa:
			la, pa = la+1, pa.next
		while pb:
			lb, pb = lb+1, pb.next
		pa, pb = headA, headB
		if la > lb:
			for i in range(la-lb):
				pa = pa.next
		elif lb > la:
			for i in range(lb-la):
				pb = pb.next
		while True:
			if pa is pb: return pa
			pa, pb = pa.next, pb.next

# 191. Number of 1 Bits
class Solution(object): # 99.49%
	def hammingWeight(self, n):
		r = 0
		while n:
			r, n = r+1, n&(n-1)
		return r

# 496. Next Greater Element I
class Solution: # 99.94%
	def nextGreaterElement(self, n1, n2):
		d, l = {}, len(n2)
		for i in range(l)[::-1]:
			if i == l-1: d[n2[i]] = None
			elif n2[i] < n2[i+1]: d[n2[i]] = n2[i+1]
			else:
				n_g = d[n2[i+1]]
				while n_g and n2[i] > n_g:
					n_g = d[n_g]
				d[n2[i]] = n_g
		return [d[_] if d[_] else -1 for _ in n1]

#————————————————19/5/17———————————————————
# 884. Uncommon Words from Two Sentences
class Solution: # 99.88%
	def uncommonFromSentences(self, A: str, B: str) -> List[str]:
		d, r = {}, []
		A, B = A.split(' '), B.split(' ')
		for sA in A:
			d[sA] = 1 if sA not in d else d[sA]+1
		for sB in B:
			d[sB] = 1 if sB not in d else d[sB]+1
		for s in d.keys():
			if d[s] == 1:
				r.append(s)
		return r

# 669. Trim a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#	def __init__(self, x):
#		self.val = x
#		self.left = None
#		self.right = None

class Solution: # 98.69%
	def trimBST(self, root: TreeNode, L: int, R: int) -> TreeNode:
		def helper(r, L, R):
			if not r: return
			if r.val < L:
				return helper(r.right, L, R)
			elif r.val > R:
				return helper(r.left, L, R)
			else:
				r.left = helper(r.left, L, R)
				r.right = helper(r.right, L, R)
				return r
		return helper(root, L, R)

#————————————————19/5/18———————————————————
# 739. Daily Temperatures
class Solution: # 93.37%
	def dailyTemperatures(self, T: List[int]) -> List[int]:
		r = [0 for t in T]
		for i in range(len(T)-1)[::-1]:
			j = i+1
			while True:
				if T[i] < T[j]:
					r[i] = j-i
					break
				if not r[j]:
					r[i] = 0
					break
				j = j+r[j]
		return r

# 575. Distribute Candies
class Solution: # 98.06%
	def distributeCandies(self, candies: List[int]) -> int:
		r, l = len(set(candies)), len(candies)
		return r if r < l // 2 else l // 2

#————————————————19/5/19———————————————————
# 463. Island Perimeter
class Solution: # 92.16%
	def islandPerimeter(self, grid: List[List[int]]) -> int:
		h, w, r = len(grid), len(grid[0]), 0
		for i in range(h):
			for j in range(w):
				if grid[i][j] == 1:
					r += 4 - (grid[i-1][j] if i > 0 else 0) - (grid[i+1][j] if i < h-1 else 0) - (grid[i][j-1] if j > 0 else 0) - (grid[i][j+1] if j < w-1 else 0)
		return r

# 442. Find All Duplicates in an Array
class Solution: # 53.38%
	def findDuplicates(self, nums: List[int]) -> List[int]:
		r = []
		for x in nums:
			if nums[abs(x)-1] < 0:
				r.append(abs(x))
			else:
				nums[abs(x)-1] *= -1
		return r

#————————————————19/5/21———————————————————
# 1047. Remove All Adjacent Duplicates In String
class Solution: # 50.80%
	def removeDuplicates(self, s: str) -> str:
		c, l = 0, len(s)
		while c < len(s)-1:
			if s[c] == s[c+1]:
				s = s[:c] + s[c+2:]
				c = c-1 if c else 0
				continue
			c += 1
		return s
class Solution: # 93.16%
	def removeDuplicates(self, s: str) -> str:
		r = []
		for c in s:
			if r and c == r[-1]:
				r.pop()
				continue
			r.append(c)
		return ''.join(r)

# 841. Keys and Rooms
class Solution: # 57.23%
	def canVisitAllRooms(self, rooms):
		visited = [False] * len(rooms)
		visited[0] = True
		stack = [0]
		while stack:
			n = stack.pop()
			for _ in rooms[n]:
				if not visited[_]:
					stack.append(_)
					visited[_] = True
		return reduce(lambda x, y: x and y, visited)

#————————————————19/5/22———————————————————
# 412. Fizz Buzz
class Solution: # 90.78%
	def fizzBuzz(self, n: int) -> List[str]:
		r = []
		for i in range(1, n+1):
			if i % 15 == 0:
				r.append("FizzBuzz")
			elif i % 3 == 0:
				r.append("Fizz")
			elif i % 5 == 0:
				r.append("Buzz")
			else:
				r.append(str(i))
		return r

# 406. Queue Reconstruction by Height
class Solution: # 88.53%
	def reconstructQueue(self, p):
		p, r, h = sorted(p, key=lambda x: (x[0], -x[1])), [0] * len(p), [i for i in range(len(p))]
		for _ in p:
			r[h[_[1]]] = _
			h.pop(_[1])
		return r

#————————————————19/5/23———————————————————
# 682. Baseball Game
class Solution: # 92.99%
	def calPoints(self, ops: List[str]) -> int:
		r = []
		for op in ops:
			if op.isdigit():
				r.append(int(op))
			elif op[0] == '-':
				r.append(-int(op[1:]))
			elif op == 'C':
				r.pop()
			elif op == 'D':
				r.append(2*r[-1])
			else:
				r.append(r[-1]+r[-2])
		return sum(r)

# 824. Goat Latin
class Solution: # 99.25%
	def toGoatLatin(self, S: str) -> str:
		words, r = S.split(' '), []
		for i, word in enumerate(words):
			if word[0] in ['a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U']:
				r.append(word + 'ma' + 'a'*(i+1))
			else:
				r.append(word[1:] + word[0] + 'ma' + 'a'*(i+1))
		return ' '.join(r)

#————————————————19/5/24———————————————————
# 566. Reshape the Matrix
class Solution: # 91.74%
	def matrixReshape(self, A, r, c):
		h, w = len(A), len(A[0])
		if h*w != r*c or (h == r and w == c):
			return A
		C = []
		for _ in A:
			for __ in _:
				C.append(__)
		return [C[i:i+c] for i in range(0, len(C), c)]

# 931. Minimum Falling Path Sum
class Solution: # 32.45%
	def minFallingPathSum(self, A):
		l = len(A)
		dp = [[0 for _ in range(l)] for _ in range(l)]

		for i in range(l):
			dp[0][i] = A[0][i]

		for i in range(1, l):
			for j in range(l):
				dp[i][j] = dp[i-1][j] + A[i][j]
				if j < l-1:
					dp[i][j] = min(dp[i][j], dp[i-1][j+1] + A[i][j])
				if j > 0:
					dp[i][j] = min(dp[i][j], dp[i-1][j-1] + A[i][j])
		return min(dp[l-1])

#————————————————19/5/25———————————————————
# 1043. Partition Array for Maximum Sum
class Solution: # 10.30%
	def maxSumAfterPartitioning(self, A, K):
		dp = [0] * (len(A)+1)
		for i in range(1, len(A)+1):
			dp[i] = max([(dp[i-j] + max(A[i-j:i])*j) for j in range(1, min(K+1, i+1))])
		return dp[-1]

# 1009. Complement of Base 10 Integer
class Solution: # 94.78%
	def bitwiseComplement(self, N):
		return sum([2**i*(b == '0') for i, b in enumerate(bin(N)[:1:-1])])

#————————————————19/5/26———————————————————
# 890. Find and Replace Pattern
class Solution: # 99.65%
	def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
		r = []
		def helper(word, pattern):
			 d, used, new_word = dict(), set(), ''
			 for i in range(len(word)):
			 	if word[i] in d:
			 		new_word += d[word[i]]
			 	else:
			 		if pattern[i] in used:
			 			break
			 		else:
			 			d[word[i]] = pattern[i]
			 			used.add(pattern[i])
			 			new_word += pattern[i]
			 return new_word == pattern

		for word in words:
			if helper(word, pattern):
				r.append(word)
		return r

# 657. Robot Return to Origin
class Solution: # 53.11%
	def judgeCircle(self, moves: str) -> bool:
		x, y = 0, 0
		for move in moves:
			if move == 'U':
				y += 1
			elif move == 'D':
				y -= 1
			elif move == 'L':
				x -= 1
			else:
				x += 1
		return (x, y) == (0, 0)

#————————————————19/6/3———————————————————
# 1051. Height Checker
class Solution: # 97.71%
	def heightChecker(self, heights: List[int]) -> int:
		l, r = sorted(heights), 0
		for i in range(len(l)):
			if l[i] != heights[i]:
				r += 1
		return r

# 950. Reveal Cards In Increasing Order
class Solution: # 87.23%
	def deckRevealedIncreasing(self, deck: List[int]) -> List[int]:
		deck.sort(reverse=True)
		r, l = [deck[0]], len(deck)
		for i in range(1, l):
			t = r.pop()
			r.insert(0, t)
			r.insert(0, deck[i])

		return r

#————————————————19/6/9———————————————————
# 461. Hamming Distance
class Solution: # 93.94%
	def hammingDistance(self, x: int, y: int) -> int:
		xb, yb, r = [*bin(x)[2:]], [*bin(y)[2:]], 0
		while xb and yb:
			if xb.pop() != yb.pop():
				r += 1
		while xb:
			if xb.pop() == '1':
				r += 1
		while yb:
			if yb.pop() == '1':
				r += 1
		return r

# 728. Self Dividing Numbers
class Solution: # 75.97%
	def selfDividingNumbers(self, left: int, right: int) -> List[int]:
		r = []
		def helper(num):
			for ch in str(num):
				if ch == '0' or num % int(ch):
					return False
			return True

		for i in range(left, right+1):
			if helper(i):
				r.append(i)
		return r

#————————————————19/7/28———————————————————
# 679. 24 Game 18.18%
class Solution:
	def judgePoint24(self, nums: List[int]) -> bool:
		if len(nums) == 1:
			return math.isclose(nums[0], 24)
		return any(self.judgePoint24([x] + rest)
			for a, b, *rest in itertools.permutations(nums)
			for x in {a+b, a-b, a*b, b and a/b})

# 1046. Last Stone Weight 96.44%
class Solution:
	def lastStoneWeight(self, stones: List[int]) -> int:
		stones.sort()
		i = len(stones)-1
		while i > 0:
			a = stones.pop()
			b = stones.pop()
			if a == b:
				i -= 2
			else:
				i -= 1
				stones.append(a-b)
				stones.sort()
		return 0 if not stones else stones[0]

#————————————————19/7/29———————————————————
# Definition for a binary tree node.
# class TreeNode:
#	def __init__(self, x):
#		self.val = x
#		self.left = None
#		self.right = None

# 671. Second Minimum Node In a Binary Tree 60.29%
class Solution:
	res = -1
	def findSecondMinimumValue(self, r: TreeNode) -> int:
		def helper(r):
			if not r:
				return
			if r.left:
				if r.left.val > r.val:
					if self.res == -1 or self.res > r.left.val:
						self.res = r.left.val
					helper(r.right)
				elif r.right.val > r.val:
					if self.res == -1 or self.res > r.right.val:
						self.res = r.right.val
					helper(r.left)
				else:
					helper(r.left)
					helper(r.right)

		helper(r)
		return self.res

# 357. Count Numbers with Unique Digits 83.93 %
class Solution:
	def countNumbersWithUniqueDigits(self, n: int) -> int:
		options = [9, 9, 8, 7, 6, 5, 4, 3, 2, 1]
		res, mul = 1, 1
		for i in range(n if n <= 10 else 10):
			mul *= options[i]
			res += mul
		return res

#————————————————19/7/31———————————————————
# 310. Minimum Height Trees 80.17% 学到了
class Solution:
	def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
		if n == 1: return [0]
		adj = [set() for i in range(n)]
		for i, j in edges:
			adj[i].add(j)
			adj[j].add(i)

		# 叶子节点
		leaves = [i for i in range(n) if len(adj[i])==1]
		while n > 2:
			n -= len(leaves)
			new_leaves = []
			for leaf in leaves:
				j = adj[leaf].pop()
				adj[j].remove(leaf)
				if len(adj[j]) == 1:
					new_leaves.append(j)
			leaves = new_leaves
		return leaves

# 598. Range Addition II 96.43%
class Solution:
	def maxCount(self, m: int, n: int, ops: List[List[int]]) -> int:
		i, j = m, n
		for a, b in ops:
			if not a or not b:
				continue
			i, j = min(i, a), min(j, b)
		return i*j

#————————————————19/8/1———————————————————
# 1002. Find Common Characters 70.29%
class Solution:
	def commonChars(self, A: List[str]) -> List[str]:
		res = collections.Counter(A[0])
		for a in A:
			res &= collections.Counter(a)
		return list(res.elements())
class Solution: # 70.29%
	def commonChars(self, A: List[str]) -> List[str]:
		r, l = A[0], len(A)
		for c in r:
			for i in range(1, l):
				if c not in A[i]:
					index = r.index(c)
					r = r[:index]+r[index+1:]
					break
				else:
					index = A[i].index(c)
					A[i] = A[i][:index]+A[i][index+1:]
		return r

# 283. Move Zeroes 5.06%
class Solution:
	def moveZeroes(self, n: List[int]) -> None:
		"""
		Do not return anything, modify n in-place instead.
		"""
		l = len(n)
		for i in range(l):
			flag = 1
			for j in range(l-i-1):
				if not n[j]:
					n[j], n[j+1] = n[j+1], n[j]
					flag = 0
			if flag:
				break
class Solution: # 61.99%
	def moveZeroes(self, nums: List[int]) -> None:
		"""
		Do not return anything, modify nums in-place instead.
		"""
		j = 0
		for n in nums:
			if n:
				nums[j] = n
				j += 1
		while j < len(nums):
			nums[j] = 0
			j += 1

#————————————————19/8/2———————————————————
# 949. Largest Time for Given Digits 63.83%
class Solution:
	def largestTimeFromDigits(self, A: List[int]) -> str:
		B = sorted(list(itertools.permutations(A)), reverse=True)
		for b in B:
			i, j, k, l = b
			hour = (i*10 + j)
			minu = (k*10 + l)

			if hour < 24 and minu < 60:
				return f"{i}{j}:{k}{l}"
		return ""

# 821. Shortest Distance to a Character 49.30%
class Solution:
	def shortestToChar(self, S: str, C: str) -> List[int]:
		indexs, res = [], []
		def helper(i, indexs):
			for j in range(len(indexs)):
				if i < indexs[j]:
					return indexs[j] - i if j == 0 else min(indexs[j]-i, i-indexs[j-1])
			return i-indexs[-1]
		for i, c in enumerate(S):
			if c == C:
				indexs.append(i)
		for i, c in enumerate(S):
			if c == C:
				res.append(0)
			else:
				res.append(helper(i, indexs))
		return res

#————————————————19/8/3———————————————————
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# 257. Binary Tree Paths 59.86%
class Solution:
	def binaryTreePaths(self, root: TreeNode) -> List[str]:
		res = []
		def helper(root, res_):
			if not root:
				return
			res_ = res_ + [str(root.val)]
			if not root.left and not root.right:
				res.append(res_)
			if root.left:
				helper(root.left, res_)
			if root.right:
				helper(root.right, res_)
		helper(root, [])
		print(res)
		return ["->".join(_) for _ in res]

# 892. Surface Area of 3D Shapes 86.49%
class Solution:
	def surfaceArea(self, grid: List[List[int]]) -> int:
		l, r = len(grid), 0
		for i in range(l):
			for j in range(l):
				if grid[i][j]:
					r += 2 + grid[i][j]*4
				if i:
					r -= min(grid[i][j], grid[i-1][j])*2
				if j:
					r -= min(grid[i][j], grid[i][j-1])*2
		return r

#————————————————19/8/4———————————————————
# 697. Degree of an Array 97.54%
class Solution:
	def findShortestSubArray(self, nums: List[int]) -> int:
		freq, start, end = collections.Counter(nums), {}, {}
		if len(freq) == len(nums):
			return 1
		max_freq = max(freq.values())
		max_freq_nums = []
		for num in freq.keys():
			if freq[num] == max_freq:
				max_freq_nums.append(num)
		for i, num in enumerate(nums):
			if num not in start:
				start[num] = i
			end[num] = i
		return min([end[max_freq_num]-start[max_freq_num]+1 for max_freq_num in max_freq_nums])
class Solution: # 81.40%
	def findShortestSubArray(self, nums: List[int]) -> int:
		freq, start, end, res = {}, {}, {}, len(nums)
		for i, num in enumerate(nums):
			if num not in start:
				start[num] = i
			end[num] = i
			freq[num] = freq.get(num, 0)+1
		if len(freq) == len(nums):
			return 1
		max_freq = max(freq.values())
		for num in freq:
			if freq[num] == max_freq:
				res = min(res, end[num]-start[num]+1)
		return res

# 680. Valid Palindrome II
class Solution: # Greedy 91.23%
	def validPalindrome(self, s: str) -> bool:
		def is_pal(i, j):
			return s[i:j] == s[i:j][::-1]
		l = len(s)
		for i in range(l//2):
			if s[i] != s[~i]:
				return is_pal(i, ~i) or is_pal(i+1, l-i)
		return True

#————————————————19/8/5———————————————————
# Repeated String Match 68.16%
class Solution:
	def repeatedStringMatch(self, A: str, B: str) -> int:
		la, lb = len(A), len(B)
		for i in range(lb//la+1 if lb%la else lb//la, lb//la+3):
			if B in A*i:
				return i
		return -1

# 523. Continuous Subarray Sum 24.21%
class Solution:
	def checkSubarraySum(self, nums: List[int], k: int) -> bool:
		l = len(nums)
		if l < 2: return False
		cur_sum = []
		for num in nums:
			cur_sum.append(cur_sum[-1]+num if cur_sum else num)
		for i in range(1, l):
			if not ((cur_sum[i] % k) if k else cur_sum[i]):
				return True
			for j in range(i-1):
				if not (((cur_sum[i]-cur_sum[j]) % k) if k else cur_sum[i]-cur_sum[j]):
					return True
		return False
class Solution: # 84.52% 牛逼
	def checkSubarraySum(self, nums: List[int], k: int) -> bool:
		cur_sum, m = 0, {0:-1}
		for i, num in enumerate(nums):
			cur_sum += num
			if k: cur_sum %= k
			pre = m.get(cur_sum, None)
			if pre is not None:
				if i-pre > 1: return True
			else:
				m[cur_sum] = i
		return False

#————————————————19/8/6———————————————————
# 965. Univalued Binary Tree 94.39%
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
	def isUnivalTree(self, root: TreeNode) -> bool:
		vals, stack = set(), [root]
		while stack:
			node = stack.pop()
			vals.add(node.val)
			if node.left:
				stack.append(node.left)
			if node.right:
				stack.append(node.right)

		return len(vals) == 1

# 665. Non-decreasing Array
class Solution: # brute force: TLE
	def checkPossibility(self, nums: List[int]) -> bool:
		if len(nums) < 3: return True

		def helper(nums):
			for i in range(len(nums)-1):
				if nums[i] > nums[i+1]:
					return False
			return True

		for i in range(len(nums)):
			if helper(nums[:i]+nums[i+1:]):
				return True
		return False
class Solution: # 57.96%
	def checkPossibility(self, nums: List[int]) -> bool:
		if len(nums) < 3: return True
		
		def helper(nums):
			for i in range(len(nums)-1):
				if nums[i] > nums[i+1]:
					return False
			return True
		for i in range(len(nums)-1):
			if nums[i] > nums[i+1]:
				dl = helper(nums[:i]) and helper(nums[i+1:]) and (nums[i-1] if i else float("-inf")) <= nums[i+1]
				dr = helper(nums[:i+1]) and helper(nums[i+2:]) and (nums[i+2] if i != len(nums)-2 else float("inf")) >= nums[i]
				return dl or dr
		return True

#————————————————19/8/7———————————————————
# 888. Fair Candy Swap
class Solution: # brute force Time Limit Exceeded
	def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
		d = sum(A) - sum(B)
		for a in A:
			for b in B:
				if a-b == d//2:
					return [a, b]
class Solution: # 22.08%
	def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
		d = (sum(A)-sum(B)) // 2
		A, B, a, b = sorted(A), sorted(B), 0, 0
		while True:
			if A[a]-B[b] == d:
				return [A[a], B[b]]
			if A[a]-B[b] < d:
				a += 1
			else:
				b += 1
class Solution: # 81.82% ...
	def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
		d = (sum(A)-sum(B)) // 2
		setA, setB = set(A), set(B)
		for a in setA:
			if a-d in setB:
				return [a, a-d]

# 82. Remove Duplicates from Sorted List II
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution: # 69.38%
	def deleteDuplicates(self, head):
		dummy = pre = ListNode(0)
		dummy.next = head
		while head and head.next:
			if head.val == head.next.val:
				while head.next and head.val == head.next.val:
					head = head.next
				head = head.next
				pre.next = head
			else:
				pre = pre.next
				head = head.next
		return dummy.next