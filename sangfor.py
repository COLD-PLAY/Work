# 1. 小球遍历
def fun1():
	line1 = input()
	K, N = (int(n) for n in line1.split(' '))
	kinds = []
	for i in range(N):
		nums = int(input())
		kinds.append(nums)

	for i in range(int(pow(10, K))):
		num = i
		for j in range(K):
			pass
			

# 2. 字符串的加密和解密
def fun2():
	line1 = input()
	line2 = int(input())
	line3 = input()
	code = 'abcdefghijklmnopqrstuvwxyz0123456789'

	res = ''
	if line2 == 1: # 加密
		for i in range(len(line3)):
			res += line1[code.index(line3[i])]

	if line2 == 0: # 解密
		for i in range(len(line3)):
			res += code[line1.index(line3[i])]
	print(res)

def func3():
	import re
	string = input()
	sub = input()
	res = re.findall(sub, string)
	if len(res) == 0:
		print(-1)
	else:
		print(len(res))

# func3()
import re
string = input()
sub = input()
res = re.findall(sub, string)
if len(res) == 0:
	print(-1)
else:
	print(len(res))

