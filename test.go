package main

import (
	"fmt"
	"strings"
)
func main() {
	// const INT_MAX = int(^uint(0) >> 1)
	// fmt.Println(INT_MAX)
	a := "233333"
	// fmt.Println(a[0:1])
	// b := make([]bool, len(a))
	// fmt.Println(b)
	strings.Replace(a, "2", "33", -1)
	fmt.Println(a)
}