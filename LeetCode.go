package main

import (
	"strings"
	"strconv"
	"fmt"
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

//————————————————19/7/11———————————————————
// 217. Contains Duplicate 98.57%
func containsDuplicate(nums []int) bool {
	ll := len(nums)
	tab := make(map[int]int, ll)
	for _, num := range nums {
		if _, ok := tab[num]; ok {
			return true
		} else {
			tab[num] = 1
		}
	}
	return false
}

// 228. Summary Ranges 100.00%
func summaryRanges(nums []int) []string {
	var res []string
	if len(nums) == 0 {
		return res
	}
	var pre, ll, i, start int
	ll, pre = len(nums), nums[0]
	for i = 1; i <= ll; i++ {
		start = i-1
		for i < ll && pre == nums[i] - 1 {
			pre, i = nums[i], i+1
		}
		if i < ll {
			pre = nums[i]
		}
		if start == i-1 {
			res = append(res, fmt.Sprintf("%d", nums[start]))
		} else {
			res = append(res, fmt.Sprintf("%d->%d", nums[start], nums[i-1]))
		}
	}
	return res
}

//————————————————19/7/12———————————————————
// 148. Sort List 20.07%
func sortList(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}
	var p = head
	var a []int
	for p != nil {
		a = append(a, p.Val)
		p = p.Next
	}
	l := len(a)
	quickSort(a, 0, l-1)
	p = head
	for _, v := range a {
		p.Val = v
		p = p.Next
	}
	return head
}
func quickSort(a []int, s, e int) {
	if s >= e {
		return
	}
	i, j, k := s, e, a[s]
	for i < j {
		for i < j && a[j] >= k {
			j--
		}
		a[i] = a[j]
		for i < j && a[i] <= k {
			i++
		}
		a[j] = a[i]
	}
	a[i] = k
	quickSort(a, s, i-1)
	quickSort(a, i+1, e)
}

//————————————————19/7/13———————————————————
// 1073. Adding Two Negabinary Numbers 78.05%
func addNegabinary(a1, a2 []int) []int {
	var c, i, j = 0, len(a1)-1, len(a2)-1
	var res []int
	for i >= 0 || j >= 0 || c != 0 {
		if i >= 0 {
			c += a1[i]
			i -= 1
		}
		if j >= 0 {
			c += a2[j]
			j -= 1
		}
		res = append(res, c & 1)
		c = -(c >> 1)
	}
	for len(res) > 1 && res[len(res)-1] == 0 {
		res = res[:len(res)-1]
	}
	i, j = 0, len(res)-1
	for i < j {
		res[i], res[j] = res[j], res[i]
		i, j = i+1, j-1
	}
	return res
}

// 837. New 21 Game 100.00% 搞不懂…
func new21Game(N int, K int, W int) float64 {
	if K == 0 || N >= K + W {
		return 1.0
	}
	var dp = make([]float64, N+1)
	dp[0] = 1.0
	var Wsum, res = 1.0, 0.0
	for i := 1; i <= N; i++ {
		dp[i] = Wsum / float64(W)
		if i < K {
			Wsum += dp[i]
		} else {
			res += dp[i]
		}
		if i >= W {
			Wsum -= dp[i-W]
		}
	}
	return res
}

//————————————————19/7/16———————————————————
// 632. Smallest Range
func smallestRange(nums [][]int) []int {
	var (
		l, i, j, INT_MAX, INT_MIN int // length, index
		mnums [][2]int // set the min&max value of arr
		s, e int
	)
	l, mnums = len(nums), make([][2]int, l)
	INT_MAX, INT_MIN = int(^uint(0) >> 1), ^int(0)
	for i = 0; i < l; i++ {
		mnums[i][0], mnums[i][1] = INT_MAX, INT_MIN
	}
	for i = 0; i < l; i++ {
		for j = 0; j < len(nums[i]); j++ {
			if mnums[i][0] > nums[i][j] {
				mnums[i][0] = nums[i][j]
			}
			if mnums[i][1] < nums[i][j] {
				mnums[i][1] = nums[i][j]
			}
		}
	}
	
}