package main

import (
	"strings"
	"strconv"
)

// Definition for singly-linked list.
type ListNode struct {
	Val int
	Next *ListNode
}

//————————————————19/7/2———————————————————
// 912. Sort an Array 5.04%
func sortArray(nums []int) []int {
	l := len(nums)
	for i := 0; i < l; i++ {
		f := true
		for j := 0; j < l - i - 1; j++ {
			if nums[j] > nums[j+1] {
				nums[j], nums[j+1] = nums[j+1], nums[j]
				f = false
			}
		}
		if f {
			break
		}
	}
	return nums
}

// 876. Middle of the Linked List 100.00%
func middleNode(head *ListNode) *ListNode {
	var (
		p *ListNode
		l, i, m int
	)
	for l, p = 0, head; p != nil; p = p.Next {
		l += 1
	}
	
	for m, i, p = l / 2, 0, head; i < m; i++ {
		p = p.Next
	}
	return p
}

//————————————————19/7/8———————————————————
// 1108. Defanging an IP Address
func defangIPaddr(address string) string {
	// return strings.Replace(address, ".", "[.]", -1)
	// Or this way
	ips := strings.Split(address, ".")
	return strings.Join(ips, "[.]")
}

// 120. Triangle # 97.06%
func minimumTotal(T [][]int) int {
	h, INT_MAX := len(T), int(^uint(0) >> 1)
	res := INT_MAX
	if h == 0 {
		return 0
	}
	for i := 1; i < h; i++ {
		for j := 0; j < i+1; j++ {
			if j == 0 {
				T[i][j] += T[i-1][j]
			} else if j == i {
				T[i][j] += T[i-1][j-1]
			} else {
				if T[i-1][j-1] > T[i-1][j] {
					T[i][j] += T[i-1][j]
				} else {
					T[i][j] += T[i-1][j-1]
				}
			}
		}
	}
	for j := 0; j < h; j++ {
		if res > T[h-1][j] {
			res = T[h-1][j]
		}
	}
	return res
}

//————————————————19/7/9———————————————————
// 139. Word Break 100.00%
func wordBreak(s string, wordDict []string) bool {
	i, j, l := 0, 0, len(s)
	var dp = make([]bool, l)
	for i = 0; i < l; i++ {
		if isInWordDicts(s[:i+1], wordDict) {
			dp[i] = true
			continue
		}
		if i == 0 {
			dp[i] = isInWordDicts(s[:1], wordDict)
		} else {
			for j = 0; j < i; j++ {
				if dp[j] {
					if isInWordDicts(s[j+1:i+1], wordDict) {
						dp[i] = true
						break
					}
				}
			}
			if j == i {
				dp[i] = false
			}
		}
	}
	return dp[len(s)-1]
}
func isInWordDicts(s string, wordDict []string) bool {
	for _, word := range wordDict {
		if s == word {
			return true
		}
	}
	return false
}

// 165. Compare Version Numbers 100.00%
func compareVersion(version1 string, version2 string) int {
	vs1, vs2 := strings.Split(version1, "."), strings.Split(version2, ".")
	l1, l2 := len(vs1), len(vs2)
	var min, i int
	if l1 < l2 {
		min = l1
	} else {
		min = l2
	}
	for i = 0; i < min; i++ {
		v1, _ := strconv.Atoi(vs1[i])
		v2, _ := strconv.Atoi(vs2[i])
		if v1 == v2 {
			continue
		} else {
			if v1 > v2 {
				return 1
			} else {
				return -1
			}
		}
	}
	if l1 == l2 {
		return 0
	} else {
		if l1 > l2 {
			for i = min; i < l1; i++ {
				v1, _ := strconv.Atoi(vs1[i])
				if v1 != 0 {
					return 1
				}
			}
		} else {
			for i = min; i < l2; i++ {
				v2, _ := strconv.Atoi(vs2[i])
				if v2 != 0 {
					return -1
				}
			}
		}
	}
	return 0
}

//————————————————19/7/10———————————————————
// 221. Maximal Square 100.00%
func maximalSquare(M [][]byte) int {
	if len(M) == 0 {
		return 0
	}
	h, w := len(M), len(M[0])
	R, Max := make([][]int, h), 0
	for i := 0; i < h; i++ {
		R[i] = make([]int, w)
	}
	for i := 0; i < h; i++ {
		if M[i][0] == '1' {
			R[i][0] = 1
		} else {
			R[i][0] = 0
		}
		if Max < R[i][0] {
			Max = R[i][0]
		}
	}
	for j := 0; j < w; j++ {
		if M[0][j] == '1' {
			R[0][j] = 1
		} else {
			R[0][j] = 0
		}
		if Max < R[0][j] {
			Max = R[0][j]
		}
	}
	for i := 1; i < h; i++ {
		for j := 1; j < w; j++ {
			if M[i][j] == '1' {
				if R[i-1][j] < R[i][j-1] {
					if R[i-1][j] < R[i-1][j-1] {
						R[i][j] = R[i-1][j] + 1
					} else {
						R[i][j] = R[i-1][j-1] + 1
					}
				} else {
					if R[i][j-1] < R[i-1][j-1] {
						R[i][j] = R[i][j-1] + 1
					} else {
						R[i][j] = R[i-1][j-1] + 1
					}
				}
				if Max < R[i][j] {
					Max = R[i][j]
				}
			}
		}
	}
	return Max * Max
}

// 459. Repeated Substring Pattern 28.87%
func repeatedSubstringPattern(s string) bool {
	i, l := 0, len(s)
	if l < 2 {
		return false
	}
	for i = 0; i < l / 2; i++ {
		if l % (i+1) == 0 {
			if isRepeatedSubstring(s[:i+1], s) {
				break
			}
		}
	}
	return i != l / 2
}
func isRepeatedSubstring(sub, s string) bool {
	return strings.Replace(s, sub, "", -1) == ""
}