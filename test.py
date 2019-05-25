# 1009. Complement of Base 10 Integer
class Solution:
	def bitwiseComplement(self, N):
		return sum([2**i*(b == '0') for i, b in enumerate(bin(N)[:1:-1])])

if __name__ == '__main__':
	import time
	s = time.time()

	S = Solution()
	r = S.bitwiseComplement(10)
	print(r)

	e = time.time()
	print('%f s' % (e - s))