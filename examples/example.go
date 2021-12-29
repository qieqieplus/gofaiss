// Package tests
// From faiss/c_api/example_c.c
package main

import (
	"fmt"
	"github.com/qieqieplus/gofaiss"
	"math"
	"math/rand"
	"time"
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
			xb[d*i+j] = rand.Float32() / math.MaxFloat32
		}
		xb[d*i] += float32(i) / 1000.0
	}

	for i := 0; i < nq; i++ {
		for j := 0; j < d; j++ {
			xq[d*i+j] = rand.Float32() / math.MaxFloat32
		}
		xq[d*i] += float32(i) / 1000.0
	}

	fmt.Println("Building an index...")
	var index *gofaiss.FaissIndex
	gofaiss.IndexFactory(&index, int32(d), "Flat", gofaiss.METRIC_L2)
	trained := gofaiss.IndexIsTrained(index) > 0
	fmt.Printf("is_trained = %v\n", trained)

	gofaiss.IndexAdd(index, int32(nb), &xb[0])
	total := gofaiss.IndexNtotal(index)
	fmt.Printf("ntotal = %v\n", total)
}

