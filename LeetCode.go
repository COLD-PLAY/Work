package main

import "fmt"

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