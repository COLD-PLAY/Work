# res = []
# for i in range(1, 8):
#     res_ = []
#     for j in range(i):
#         res_.append(1 if j == 0 or j == i - 1 else res[-1][j] + res[-1][j - 1])
#     res.append(res_)
# print(res)
# res = [[1, 2, 3], [1, 2, 3], [2, 3, 4]]
# res = [1, 2, 3, 4, 1]
# res = list(set(res))
# print(res)
def fourSum(nums, target: int):
	nums.sort()
	length = len(nums)
	res = []
	for i in range(length - 3):
		if i > 0 and nums[i] == nums[i - 1]: continue
		for j in range(i + 1, length - 2):
			if j > i + 1 and nums[j] == nums[j - 1]: continue
			l = j + 1
			r = length - 1
			while l < r:
				if nums[i] + nums[j] + nums[l] + nums[r] < target:
					l += 1
				elif nums[i] + nums[j] + nums[l] + nums[r] > target:
					r -= 1
				else:
					res.append([nums[i], nums[j], nums[l], nums[r]])
					while l < r and nums[l] == nums[l + 1]: l += 1
					while l < r and nums[r] == nums[r - 1]: r -= 1
					l += 1
					r -= 1

	return res

res = fourSum([-491,-486,-481,-479,-463,-453,-405,-393,-388,-385,-381,-380,-347,-340,-334,-333,-326,-325,-321,-321,-318,-317,-269,-261,-252,-241,-233,-231,-209,-203,-203,-196,-187,-181,-169,-158,-138,-120,-111,-92,-86,-74,-33,-14,-13,-10,-5,-1,8,32,48,73,80,82,84,85,92,134,153,163,192,199,199,206,206,217,232,249,258,326,329,341,343,345,363,378,399,409,428,431,447,449,455,476,493], 2328)
print(res)