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

if __name__ == "__main__":
	test3()