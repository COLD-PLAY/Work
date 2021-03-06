package main

import (
	"strings"
	"strconv"
	"fmt"
	"sort"
	"encoding/json"
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

//————————————————19/7/19———————————————————
// 98. Validate Binary Search Tree
// Definition for a binary tree node.
type TreeNode struct { // 98.22%
	Val int
	Left *TreeNode
	Right *TreeNode
}
var orders []int
func inOrder(root *TreeNode) {
	if root == nil {
		return
	}
	if root.Left != nil {
		inOrder(root.Left)
	}
	orders = append(orders, root.Val)
	if root.Right != nil {
		inOrder(root.Right)
	}
}
func isValidBST(root *TreeNode) bool {
	orders = orders[:0]
	inOrder(root)
	for i := 0; i < len(orders) - 1; i++ {
		if orders[i] >= orders[i+1] {
			return false
		}
	}
	return true
}

// 451. Sort Characters By Frequency 17.92%
func frequencySort(s string) string {
	hT := make(map[string]int)
	var res = ""
	for i, _ := range s {
		ss := s[i:i+1]
		if _, ok := hT[ss]; ok {
			hT[ss] += 1
		} else {
			hT[ss] = 1
		}
	}
	l := len(hT)
	for i := 0; i < l; i++ {
		max, key := ^int(^uint(0) >> 1), ""
		for k, v := range hT {
			if max < v {
				max, key = v, k
			}
		}
		for j := 0; j < max; j++ {
			res += key
		}
		delete(hT, key)
	}
	return res
}

//————————————————19/7/20———————————————————
// 594. Longest Harmonious Subsequence 100.00%
func findLHS(nums []int) int {
	hT, res := make(map[int]int), 0
	for _, num := range nums {
		if _, ok := hT[num]; ok {
			hT[num] += 1
		} else {
			hT[num] = 1
		}
	}
	for k, v := range hT {
		if _, ok := hT[k+1]; ok {
			l := v + hT[k+1]
			if l > res {
				res = l
			}
		}
	}
	return res
}

// 386. Lexicographical Numbers 33.33%
func dfs(cur, n int, res *[]int) {
	if cur > n {
		return
	}
	*res = append(*res, cur)
	for i := 0; i < 10; i++ {
		if 10 * cur + i > n {
			return
		}
		dfs(10 * cur + i, n, res)
	}
}
func lexicalOrder(n int) []int {
	var res []int
	for i := 1; i < 10; i++ {
		dfs(i, n, &res)
	}
	return res
}

//————————————————19/7/22———————————————————
// 455. Assign Cookies 94.68%
func findContentChildren(g []int, s []int) int {
	sort.Ints(g)
	sort.Ints(s)
	res := 0
	for len(g) != 0 && len(s) != 0 {
		if g[0] > s[0]{

		} else {
			res += 1
			g = g[1:]
		}
		s = s[1:]
	}
	return res
}

// 748. Shortest Completing Word 6.85%
func isCompletingWord(char_frequency map[rune]int, word string) bool {
	for _, ch := range word {
		if ch < 91 {
			ch += 32
		}
		if _, ok := char_frequency[ch]; ok {
			char_frequency[ch] -= 1
		}
	}
	for _, v := range char_frequency {
		if v > 0 {
			return false
		}
	}
	return true
}
func shortestCompletingWord(licensePlate string, words []string) string {
	var (
		res = ""
		char_frequency = make(map[rune]int)
	)
	for _, ch := range licensePlate {
		if (64 < ch && ch < 91) || (96 < ch && ch < 123) {
			if ch < 91 {
				ch += 32
			}
			if _, ok := char_frequency[ch]; ok {
				char_frequency[ch] += 1
			} else {
				char_frequency[ch] = 1
			}
		}
	}
	char_frequency_json, err := json.Marshal(char_frequency)
	if err != nil {

	}
	for _, word := range words {
		char_frequency__ := make(map[rune]int)
		err := json.Unmarshal(char_frequency_json, &char_frequency__)
		fmt.Println(char_frequency__)
		if err != nil {
		
		}
		if isCompletingWord(char_frequency__, word) {
			if len(res) == 0 {
				res = word
			} else {
				if len(res) > len(word) {
					res = word
				}
			}
		}
	}
	return res
}

//————————————————19/7/23———————————————————
// 1078. Occurrences After Bigram 100.00%
func findOcurrences(text string, first string, second string) []string {
	words := strings.Split(text, " ")
	var l = len(words)
	var res []string
	for i := 0; i < l - 2; i++ {
		if words[i] == first {
			if words[i+1] == second {
				res = append(res, words[i+2])
			}
		}
	}
	return res
}

// 299. Bulls and Cows 36.21%
func getHint(secret string, guess string) string {
	A, B, l := 0, 0, len(secret)
	for i := 0; i < l; i++ {
		if secret[i] == guess[i] {
			A += 1
		}
	}
	secret_ch := strings.Split(secret, "")
	guess_ch := strings.Split(guess, "")
	sort.Strings(secret_ch)
	sort.Strings(guess_ch)

	for i, j := 0, 0; i < l && j < l; {
		if secret_ch[i] == guess_ch[j] {
			i, j, B = i+1, j+1, B+1
		} else if secret_ch[i] < guess_ch[j] {
			i += 1
		} else {
			j += 1
		}
	}
	if B-A >= 0 {
		B -= A
	}
	res := strconv.Itoa(A) + "A" + strconv.Itoa(B) + "B"
	return res
}

//————————————————19/7/24———————————————————
// 326. Power of Three 67.06%
func isPowerOfThree(n int) bool {
	if n == 3 || n == 1 {
		return true
	} else if n % 3 != 0 || n == 0 {
		return false
	}
	return isPowerOfThree(n / 3)
}

// 609. Find Duplicate File in System 96.88%
func findDuplicate(paths []string) [][]string {
	var (
		res [][]string
		content2path = make(map[string][]string)
	)
	for _, path := range paths {
		d_files := strings.Split(path, " ")
		dir, files := d_files[0], d_files[1:]
		for _, file := range files {
			file_content := strings.Split(file, "(")
			file, content := file_content[0], file_content[1]
			content2path[content] = append(content2path[content], dir + "/" + file)
		}
	}
	for _, v := range content2path {
		if len(v) > 1 {
			res = append(res, v)
		}
	}
	return res
}

//————————————————19/7/27———————————————————
// 1011. Capacity To Ship Packages Within D Days 78.43%
func possible(weights []int, D, capacity int) bool {
	if D == 0 {
		return len(weights) == 0
	} else if len(weights) == 0 {
		return true
	}
	sum, next, lw := 0, 0, len(weights)
	for i, weight := range weights {
		sum, next = sum + weight, i
		if sum > capacity {
			break
		} else if i == lw - 1 { // 最后一个package
			next = lw
		}
	}
	return possible(weights[next:], D-1, capacity)
}
func shipWithinDays(weights []int, D int) int {
	l, r := 1, 0
	for _, v := range weights {
		r += v
	}
	for l < r {
		m := (l + r) / 2
		if possible(weights, D, m) {
			r = m
		} else {
			l = m + 1
		}
	}
	return r
}

// 893. Groups of Special-Equivalent Strings 50.00%
func numSpecialEquivGroups(A []string) int {
	res := make(map[string]int)
	for _, a := range A {
		even, odd := "", ""
		for i := 0; i < len(a); i += 2 {
			even += a[i:i+1]
		}
		for i := 1; i < len(a); i += 2 {
			odd += a[i:i+1]
		}
		evens := strings.Split(even, "")
		sort.Strings(evens)
		odds := strings.Split(odd, "")
		sort.Strings(odds)
		even, odd = "", ""
		for _, e := range evens {
			even += e
		}
		for _, o := range odds {
			odd += o
		}
		if _, ok := res[even+odd]; ok {
			res[even+odd] += 1
		} else {
			res[even+odd] = 1
		}
	}
	return len(res)
}