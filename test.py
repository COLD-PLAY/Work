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
	
if __name__ == "__main__":
	test6()