package main

import (
	"fmt"
	// "strings"
)
func main() {
	content2path := make(map[string][]string)
	if _, ok := content2path["2333"]; ok {
		fmt.Println("2333")
	} else {
		content2path["2333"] = append(content2path["2333"], "liaozhou")
	}

	for k, v := range content2path {
		fmt.Println(k)
		fmt.Println(v)
	}
}