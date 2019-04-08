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
		min_len = min(length)
		for i in range(min_len):
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
				res_.append(1 if j == 0 or j == i - 1 else res[-1][j] + res[-1][j - 1])
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
				if j > i + 1 and nums[j] == nums[j - 1]: continue
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