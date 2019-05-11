def fourSum(nums, target):
	nums.sort()
	length = len(nums)
	res = []
	for i in range(length - 3):
		if i > 0 and nums[i] == nums[i - 1]: continue
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

# res = fourSum([-491,-486,-481,-479,-463,-453,-405,-393,-388,-385,-381,-380,-347,-340,-334,-333,-326,-325,-321,-321,-318,-317,-269,-261,-252,-241,-233,-231,-209,-203,-203,-196,-187,-181,-169,-158,-138,-120,-111,-92,-86,-74,-33,-14,-13,-10,-5,-1,8,32,48,73,80,82,84,85,92,134,153,163,192,199,199,206,206,217,232,249,258,326,329,341,343,345,363,378,399,409,428,431,447,449,455,476,493], 2328)
# print(res)
def isValidSudoku(board):
	board_ = []
	for i in range(9):
		board__ = []
		for j in range(9):
			if board[i][j] != '.':
				board__.append(board[i][j])

		board_.append(board__)
	for i in range(9):
		board__ = []
		for j in range(9):
			if board[j][i] != '.':
				board__.append(board[j][i])

		board_.append(board__)
	for i in range(9):
		board__ = []
		for j in range(9):
			if board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3] != '.':
				board__.append(board[i // 3 * 3 + j // 3][i % 3 * 3 + j % 3])

		board_.append(board__)
	for board__ in board_:
		# if len(board__) != len(set(board__)):
		# 	return False
		print(board__)
	return True

# board = [
#   ["5","3",".",".","7",".",".",".","."],
#   ["6",".",".","1","9","5",".",".","."],
#   [".","9","8",".",".",".",".","6","."],
#   ["8",".",".",".","6",".",".",".","3"],
#   ["4",".",".","8",".","3",".",".","1"],
#   ["7",".",".",".","2",".",".",".","6"],
#   [".","6",".",".",".",".","2","8","."],
#   [".",".",".","4","1","9",".",".","5"],
#   [".",".",".",".","8",".",".","7","9"]
# ]
# isValidSudoku(board)
# b = 1
# a = 2
# a += 1 if b else 0
# print(a)

def helper(res, a, b, c):
	print('res: %s' % res)
	if len(a) == 0 and len(b) == 0:
		return ('1' if c else '') + res
	
	d = (c + (1 if len(a) and a[-1] == '1' else 0) + (1 if len(b) and b[-1] == '1' else 0)) % 2
	c = (c + (1 if len(a) and a[-1] == '1' else 0) + (1 if len(b) and b[-1] == '1' else 0)) // 2
	return helper(('1' if d else '0') + res, a[:-1], b[:-1], c)
	
def maxProfit(prices):
	"""
	:type prices: List[int]
	:rtype: int
	"""
	min = prices[0]
	max = 0
	for price in prices:
		if price - min > max: max = price - min
		if price < min: min = price
	return max

def uniquePaths(m, n):
	"""
	:type m: int
	:type n: int
	:rtype: int
	"""
	if m == 1 or n == 1: return 1
	dp = [[1 for i in range(m)] for j in range(n)]
	for i in range(1, n):
		for j in range(1, m):
			dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
	return dp[n - 1][m - 1]

def nextPermutation(nums):
	"""
	:type nums: List[int]
	:rtype: None Do not return anything, modify nums in-place instead.
	"""
	j, l = len(nums) - 1, len(nums)
	while j:
		if nums[j] > nums[j - 1]:
			break
		j -= 1
	if j:
		if nums[-1] > nums[j - 1]:
			nums[-1], nums[j - 1] = nums[j - 1], nums[-1]
		else:
			for i in range(j, l):
				if nums[i] > nums[j - 1] and nums[i + 1] <= nums[j - 1]:
					nums[i], nums[j - 1] = nums[j - 1], nums[i]
		for i in range((l - j) // 2):
			nums[i + j], nums[l - i - 1] = nums[l - i - 1], nums[i + j]
	else:
		nums.sort()

def numDecodings(s):
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

def minCostClimbingStairs(cost):
	"""
	:type cost: List[int]
	:rtype: int
	"""
	if not cost: return 0
	if len(cost) == 1: return cost[0]
	p = c = 0
	for _ in cost:
		c, p = min(_ + c, _ + p), c
	return min(c, p)

def numTrees(n):
	"""
	:type n: int
	:rtype: int
	"""
	if n < 2: return n
	res = [0] * (n + 1)
	res[0] = res[1] = 1
	for i in range(2, n + 1):
		for j in range(i):
			res[i] += res[j] * res[i - j - 1]
	return res[-1]


def combinationSum2(candidates, target):
	"""
	:type candidates: List[int]
	:type target: int
	:rtype: List[List[int]]
	"""
	res = []
	def helper(c, s, t, r):
		if t == 0:
			res.append(r)
			return
		for i in range(s, len(c)):
			if i > s and c[i] == c[i - 1]: continue 
			if c[i] > t: break
			helper(c, i + 1, t - c[i], r + [c[i]])

	helper(sorted(candidates), 0, target, [])
	return res

def canJump(nums):
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
		print(n)
		c = n

	return False

def multiply(num1, num2):
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

def permuteUnique1(nums):
	"""
	:type nums: List[int]
	:rtype: List[List[int]]
	"""
	ans = []
	def helper(nums, res):
		if len(nums) == 0:
			if res not in ans:
				ans.append(res)
			return
		for i in range(len(nums)):
			helper(nums[:i] + nums[i + 1:], res + [nums[i]])

	helper(nums, [])
	return ans

def permuteUnique2(nums):
	"""
	:type nums: List[int]
	:rtype: List[List[int]]
	"""
	if not nums: return []
	def helper(nums):
		if not nums: return [[]]
		return [_ + [nums[i]] for i in range(len(nums)) for _ in helper(nums[:i] + nums[i+1:])]
	return [list(_) for _ in set([tuple(_) for _ in helper(nums)])]

def permuteUnique3(nums):
	perms = [[]]
	for i, n in enumerate(nums):
		perms = [p[:i] + [n] + p[i:] for p in perms for i in range(len(p) + 1)]
	return [list(p) for p in set([tuple(p) for p in perms])]

def groupAnagrams(strs):
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

def generateMatrix(n):
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
	print(res)

def getPermutation(n, k):
	def helper(r, k):
		if k == 1: return r
		f = reduce(lambda x,y: x*y, [i for i in range(1, len(r))])
		a, b = (k-1)//f, k % f if k % f else f
		return r[a] + helper(r[:a] + r[a+1:], b)
	return helper(''.join([str(i) for i in range(1, n + 1)]), k)

class Solution:
	def uniquePathsWithObstacles(self, A):
		if A[0][0] or A[-1][-1] or not A: return 0
		h, w = len(A), len(A[0])
		m = [[1 for j in range(w)] for i in range(h)]
		for i in range(h):
			for j in range(w):
				if i == 0 and j == 0: m[i][j] == 1
				else: m[i][j] = 0 if A[i][j] else (m[i - 1][j] if i > 0 else 0) + (m[i][j - 1] if j > 0 else 0)

		return m[-1][-1]

class Solution:
	def combine(self, n, k):
		self.res = []
		def helper(r, s, n, k):
			if k == 0:
				self.res.append(r)
				return
			for i in range(s, n + 1):
				helper(r + [i], i + 1, n, k - 1)

		helper([], 1, n, k)
		return self.res

class Solution:
	def searchMatrix(self, A, t):
		if not A: return False
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
		if x == -1:
			print('x error')
			return False
		while l < r:
			m = (l + r) // 2
			if A[x][m] == t:
				y = m
				break
			elif A[x][m] < t: l = m + 1
			else: r = m - 1
		if y == -1:
			print('y error')
			return False
		return True

class Solution:
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

class Solution:
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

class Solution:
	def grayCode(self, n):
		r = [0]
		for i in range(2 ** n - 1):
			for i in range(n):
				if r[-1]^(2 ** i) not in r:
					r.append(r[-1]^(2 ** i))
					break
		return r

class Solution:
	def restoreIpAddresses(self, s):
		l, a = len(s), []
		if l > 12 or l < 4: return a
		def helper(r, s, k):
			if not k:
				if not s:
					a.append(r[1:])
				return
			if s: helper(r + '.' + s[0], s[1:], k - 1)
			if len(s) > 1 and s[0] != '0': helper(r + '.' + s[:2], s[2:], k - 1)
			if len(s) > 2 and s[0] != '0' and int(s[:3]) < 256: helper(r + '.' + s[:3], s[3:], k - 1)
		helper('', s, 4)
		return a

# 868. Binary Gap
class Solution:
	def binaryGap(self, N):
		r = bin(N)[2:].split('1')[1:-1]
		if not r: return 0
		return max(map(len, r))+1

class Solution:
	def advantageCount(self, A, B):
		r = [0 for _ in A]
		A.sort()
		for b, i in sorted((b, i) for i, b in enumerate(B))[::-1]:
			if A[-1] > b:
				r[i] = A.pop()
			else:
				r[i] = A.pop(0)
		return r

class Solution:
	def getRow(self, rowIndex):
		res = [1]
		for i in range(1, rowIndex+1):
			res.append(1)
			for j in range(1, i)[::-1]:
				res[j] = res[j] + res[j-1]
		return res

if __name__ == '__main__':
	import time
	s = time.time()

	S = Solution()
	print(S.getRow(5))

	e = time.time()
	print('%f s' % (e - s))