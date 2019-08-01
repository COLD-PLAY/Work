def test1(A):
	A = [''.join(sorted(_)) for _ in A]
	print(A)

def test2():
	i, j = 0, 1
	j, i = i, j+1
	print(j, i)

if __name__ == "__main__":
	test2()