# import xlrd, xlwt
# readbook = xlrd.open_workbook('计算机学院团队人员情况（201809）.xls')
# sheet = readbook.sheet_by_index(0)
# teachers1 = []
# for i in range(143, 166):
# 	teacher = sheet.cell_value(i, 3)
# 	if len(teacher) != 0:
# 		teachers1.append(teacher)

# teachers2 = []
# for i in range(144):
# 	teacher = sheet.cell_value(i, 8)
# 	teachers2.append(teacher)

# for i in range(len(teachers1)):
# 	if teachers1[i] in teachers2:
# 		print(teachers1[i])

def solution1():
	string = input()
	length = len(string)
	l = []
	for i in range(length):
		# print(string[length-i-1], end='')
		l.append(string[length-i-1])

	return ''.join(l)

for i in range(30000, 100000):
	if (i // 10000) % 3 == 0 and (i // 1000) % 5 == 0 and (i // 100) % 7 == 0 and (i // 10) % 6 == 0 and i % 14 == 0:
		print(i)
		break