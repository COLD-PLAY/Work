class Solution:
	def isSubsequence(self, s: str, t: str) -> bool:
		if not s: return True
		if s[0] not in t: return False
		return self.isSubsequence(s[1:], t[t.index(s[0])+1:])

if __name__ == '__main__':
	import time
	s = time.time()

	a = "leeeeetcode"
	b = 'yyyyyylyyyyyyeyyyyyeyyyyyyyyyyyyyyyyytyyyyycyyyoyyyyyyyydyyyyyyyeyyyy'

	S = Solution()
	print(S.isSubsequence(a, b))

	e = time.time()
	print('%f s' % (e - s))