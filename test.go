package main

import (
	"fmt"
	// "strings"
)
func main() {
	var a = []int{4,19,14,5,-3,1,8,5,11,15}
	quickSort(a, 0, 9)
	for _, v := range a {
		fmt.Println(v)
	}
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