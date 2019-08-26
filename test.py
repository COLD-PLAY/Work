def test1(A):
	A = [''.join(sorted(_)) for _ in A]
	print(A)

def test2():
	i, j = 0, 1
	j, i = i, j+1
	print(j, i)

def test3():
	a = "abbcc"
	print(a)
	print(a[0:-1][::-1])

def test4():
	a = [23,3]
	print(a)
	print(a.pop())
	print(a)

def test5():
	print(float('inf') > 10000000000)

def test6():
	a = {}
	a[0] = a.get(0, 0)+1
	print(a[0])

def test7():
	a = 10
	print(bin(a)[2:].count('1'))

def test8():
	from functools import reduce
	primes, res = [2, 3, 5, 7, 11, 13, 17, 19], 1
	res = reduce(lambda x, y: x*y, primes)
	print(res)

def test9():
	a = [[1,0], [2,0]]
	b = a.pop([1,0])
	print(b, a)

def test10():
	a = 65
	print(chr(a))

def test11():
	a = "abcde"
	print(a.index('adbc'))

def test12():
	a = [213,1,123]
	b = a.pop(2)
	print(a, b)

def test13():
	a = set([1,2,3,2])
	b = set([5,2,4,2])
	print(a&b)

if __name__ == "__main__":
	test13()