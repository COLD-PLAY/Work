class Solution:
	def largestNumber(self, nums):
		r = ''.join(sorted(map(str, nums), lambda x, y: [1, -1][x+y > y+x]))
		return r if r[0] != '0' else '0'
if __name__ == '__main__':
	import time
	s = time.time()

	S = Solution()
	print(S.largestNumber([824,938,1399,5607,6973,5703,9609,4398,8247]))

	e = time.time()
	print('%f s' % (e - s))