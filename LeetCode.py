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
				res_.append(1 if j == 0 or j == i - 1 else res[-1][j] + res[-1][	])
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