package main

import (
	"fmt"
	"strings"
)

func main() {
	fmt.Println("Hello World!")
}

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