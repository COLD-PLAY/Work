class Solution:
	def reconstructQueue(self, p):
		p, r, h = sorted(p, key=lambda x: (x[0], -x[1])), [0] * len(p), [i for i in range(len(p))]
		for _ in p:
			r[h[_[1]]] = _
			h.pop(_[1])
		return r

if __name__ == '__main__':
	import time
	s = time.time()

	S = Solution()
	a = '233' + 'i'*2
	print(a)

	e = time.time()
	print('%f s' % (e - s))