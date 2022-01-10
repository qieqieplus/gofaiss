// Package tests
// From faiss/c_api/example_c.c
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/qieqieplus/gofaiss"
)

func main() {
	seed := time.Now().UnixNano()
	rand.Seed(seed)
	fmt.Println("Generating some data...")
	d := 128
	nb := 100000
	nq := 10000
	xb := make([]float32, d*nb)
	xq := make([]float32, d*nq)

	for i := 0; i < nb; i++ {
		for j := 0; j < d; j++ {
			xb[d*i+j] = rand.Float32()
		}
		xb[d*i] += float32(i / 1000)
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[d*i+j] = rand.Float32()
		}
		xq[d*i] += float32(i / 1000)
	}

	fmt.Println("Building an index...")
	var index *gofaiss.FaissIndex
	gofaiss.IndexFactory(&index, int32(d), "Flat", gofaiss.METRIC_L2)
	trained := gofaiss.IndexIsTrained(index) > 0
	fmt.Printf("is_trained = %v\n", trained)

	gofaiss.IndexAdd(index, nb, &xb[0])
	total := gofaiss.IndexNtotal(index)
	fmt.Printf("ntotal = %v\n", total)

	{
		n, k := 5, 5
		I := make([]int, k*n)
		D := make([]float32, k*n)
		gofaiss.IndexSearch(index, n, &xb[0], k, &D[0], &I[0])
		fmt.Printf("I=\n")
		for i := 0; i < n; i++ {
			for j := 0; j < k; j++ {
				fmt.Printf("%5d (d=%2.3f)  ", I[i*k+j], D[i*k+j])
			}
			fmt.Println("")
		}
	}

	{
		k := 5
		I := make([]int, k*nq)
		D := make([]float32, k*nq)
		gofaiss.IndexSearch(index, nq, &xq[0], k, &D[0], &I[0])
		fmt.Printf("I=\n")
		for i := 0; i < nq; i++ {
			for j := 0; j < k; j++ {
				fmt.Printf("%5d (d=%2.3f)  ", I[i*k+j], D[i*k+j])
			}
			fmt.Println("")
		}
	}

	fmt.Println("Freeing index...")
	gofaiss.IndexFree(index)
	fmt.Println("Done.")
}
