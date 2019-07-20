package main

import (
	"fmt"
	// "strings"
)
func main() {
	INT_MIN := ^int(^uint(0) >> 1)
	fmt.Println(INT_MIN)
}