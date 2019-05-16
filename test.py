class Solution:
	def largestNumber(self, nums):
		r = ''.join(sorted(map(str, nums), lambda x, y: [1, -1][x+y > y+x]))
		return r if r[0] != '0' else '0'
if __name__ == '__main__':
	import time
	s = time.time()

	S = Solution()

	e = time.time()
	print('%f s' % (e - s))