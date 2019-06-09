# 1009. Complement of Base 10 Integer
class Solution:
	def bitwiseComplement(self, N):
		return sum([2**i*(b == '0') for i, b in enumerate(bin(N)[:1:-1])])

if __name__ == '__main__':
	import time
	s = time.time()

	x, y = 1, 4
	xb, yb = [*bin(x)[2:]], [*bin(y)[2:]]
	print(yb)

	e = time.time()
	print('%f s' % (e - s))