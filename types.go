// WARNING: This file has automatically been generated on Mon, 10 Jan 2022 11:46:52 CST.
// Code generated by https://git.io/c-for-go. DO NOT EDIT.

package gofaiss

/*
#cgo pkg-config: faiss
#include "faiss.h"
#include <stdlib.h>
#include "cgo_helpers.h"
*/
import "C"

// FaissParameterRange as declared in c_api/AutoTune_c.h:22
type FaissParameterRange C.FaissParameterRange

// FaissParameterSpace as declared in c_api/AutoTune_c.h:32
type FaissParameterSpace C.FaissParameterSpace

// FaissRangeSearchResult as declared in c_api/Index_c.h:22
type FaissRangeSearchResult C.FaissRangeSearchResult

// FaissIDSelector as declared in c_api/Index_c.h:25
type FaissIDSelector C.FaissIDSelector

// FaissIndex as declared in c_api/Index_c.h:43
type FaissIndex C.FaissIndex

// Idx type as declared in c_api/faiss_c.h:18
type Idx int

// Component type as declared in c_api/faiss_c.h:20
type Component float32

// Distance type as declared in c_api/faiss_c.h:21
type Distance float32

// FaissClusteringParameters as declared in c_api/Clustering_c.h:40
type FaissClusteringParameters struct {
	Niter                int32
	Nredo                int32
	Verbose              int32
	Spherical            int32
	IntCentroids         int32
	UpdateIndex          int32
	FrozenCentroids      int32
	MinPointsPerCentroid int32
	MaxPointsPerCentroid int32
	Seed                 int32
	DecodeBlockSize      uint
	ref9c9731c3          *C.FaissClusteringParameters
	allocs9c9731c3       interface{}
}

// FaissClustering as declared in c_api/Clustering_c.h:59
type FaissClustering C.FaissClustering

// FaissClusteringIterationStats as declared in c_api/Clustering_c.h:81
type FaissClusteringIterationStats C.FaissClusteringIterationStats

// FaissIndexFlat as declared in c_api/IndexFlat_c.h:25
type FaissIndexFlat C.FaissIndexFlat

// FaissIndexFlatIP as declared in c_api/IndexFlat_c.h:71
type FaissIndexFlatIP C.FaissIndexFlatIP

// FaissIndexFlatL2 as declared in c_api/IndexFlat_c.h:81
type FaissIndexFlatL2 C.FaissIndexFlatL2

// FaissIndexRefineFlat as declared in c_api/IndexFlat_c.h:95
type FaissIndexRefineFlat C.FaissIndexRefineFlat

// FaissIndexFlat1D as declared in c_api/IndexFlat_c.h:114
type FaissIndexFlat1D C.FaissIndexFlat1D

// FaissIndexIVFFlat as declared in c_api/IndexIVFFlat_c.h:26
type FaissIndexIVFFlat C.FaissIndexIVFFlat

// FaissIndexIVF as declared in c_api/IndexIVF_c.h:39
type FaissIndexIVF C.FaissIndexIVF

// FaissIndexIVFStats as declared in c_api/IndexIVF_c.h:148
type FaissIndexIVFStats struct {
	Nq               uint
	Nlist            uint
	Ndis             uint
	NheapUpdates     uint
	QuantizationTime float64
	SearchTime       float64
	refda0d0668      *C.FaissIndexIVFStats
	allocsda0d0668   interface{}
}

// FaissIndexLSH as declared in c_api/IndexLSH_c.h:23
type FaissIndexLSH C.FaissIndexLSH

// FaissIndexPreTransform as declared in c_api/IndexPreTransform_c.h:24
type FaissIndexPreTransform C.FaissIndexPreTransform

// FaissVectorTransform as declared in c_api/VectorTransform_c.h:25
type FaissVectorTransform C.FaissVectorTransform

// FaissLinearTransform as declared in c_api/VectorTransform_c.h:76
type FaissLinearTransform C.FaissLinearTransform

// FaissRandomRotationMatrix as declared in c_api/VectorTransform_c.h:96
type FaissRandomRotationMatrix C.FaissRandomRotationMatrix

// FaissPCAMatrix as declared in c_api/VectorTransform_c.h:104
type FaissPCAMatrix C.FaissPCAMatrix

// FaissITQMatrix as declared in c_api/VectorTransform_c.h:120
type FaissITQMatrix C.FaissITQMatrix

// FaissITQTransform as declared in c_api/VectorTransform_c.h:125
type FaissITQTransform C.FaissITQTransform

// FaissOPQMatrix as declared in c_api/VectorTransform_c.h:137
type FaissOPQMatrix C.FaissOPQMatrix

// FaissRemapDimensionsTransform as declared in c_api/VectorTransform_c.h:146
type FaissRemapDimensionsTransform C.FaissRemapDimensionsTransform

// FaissNormalizationTransform as declared in c_api/VectorTransform_c.h:155
type FaissNormalizationTransform C.FaissNormalizationTransform

// FaissCenteringTransform as declared in c_api/VectorTransform_c.h:165
type FaissCenteringTransform C.FaissCenteringTransform

// FaissIndexReplicas as declared in c_api/IndexReplicas_c.h:23
type FaissIndexReplicas C.FaissIndexReplicas

// FaissIndexScalarQuantizer as declared in c_api/IndexScalarQuantizer_c.h:35
type FaissIndexScalarQuantizer C.FaissIndexScalarQuantizer

// FaissIndexIVFScalarQuantizer as declared in c_api/IndexScalarQuantizer_c.h:50
type FaissIndexIVFScalarQuantizer C.FaissIndexIVFScalarQuantizer

// FaissIndexShards as declared in c_api/IndexShards_c.h:23
type FaissIndexShards C.FaissIndexShards

// FaissIndexIDMap as declared in c_api/MetaIndexes_c.h:22
type FaissIndexIDMap C.FaissIndexIDMap

// FaissIndexIDMap2 as declared in c_api/MetaIndexes_c.h:59
type FaissIndexIDMap2 C.FaissIndexIDMap2

// FaissIDSelectorRange as declared in impl/AuxIndexStructures_c.h:58
type FaissIDSelectorRange C.FaissIDSelectorRange

// FaissIDSelectorBatch as declared in impl/AuxIndexStructures_c.h:75
type FaissIDSelectorBatch C.FaissIDSelectorBatch

// FaissBufferList as declared in impl/AuxIndexStructures_c.h:89
type FaissBufferList C.FaissBufferList

// FaissBuffer as declared in impl/AuxIndexStructures_c.h:98
type FaissBuffer struct {
	Ids            []int
	Dis            []float32
	refe3f43253    *C.FaissBuffer
	allocse3f43253 interface{}
}

// FaissRangeSearchPartialResult as declared in impl/AuxIndexStructures_c.h:116
type FaissRangeSearchPartialResult C.FaissRangeSearchPartialResult

// FaissRangeQueryResult as declared in impl/AuxIndexStructures_c.h:119
type FaissRangeQueryResult C.FaissRangeQueryResult

// FaissDistanceComputer as declared in impl/AuxIndexStructures_c.h:142
type FaissDistanceComputer C.FaissDistanceComputer
