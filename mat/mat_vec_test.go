package mat_test

import (
	"testing"

	"github.com/mrfuxi/neural/mat"
	"github.com/stretchr/testify/assert"
)

type vectorExample struct {
	src      []float64
	dst      []float64
	expected []float64
}

type matrixExample struct {
	src      [][]float64
	dst      [][]float64
	expected [][]float64
}

func TestSumVector(t *testing.T) {
	examples := []vectorExample{
		{ // All equal
			dst:      []float64{-1, -1, -1, -1, -1},
			src:      []float64{0, 1, 2, 3, 4},
			expected: []float64{-1, 0, 1, 2, 3},
		},
		{ // src shorter
			dst:      []float64{-1, -1, -1, -1, -1},
			src:      []float64{0, 1},
			expected: []float64{-1, 0, -1, -1, -1},
		},
	}

	for _, example := range examples {
		mat.SumVector(example.dst, example.src)
		assert.Equal(t, example.expected, example.dst)
	}
}

func TestSumVectorPanic(t *testing.T) {
	examples := []vectorExample{
		{ // dst shorter
			dst: []float64{-1, -1},
			src: []float64{0, 1, 2, 3, 4},
		},
	}

	for _, example := range examples {
		assert.Panics(t, func() {
			mat.SumVector(example.dst, example.src)
		})
	}
}

func TestSumMatrix(t *testing.T) {
	examples := []matrixExample{
		{ // All rows same size
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1},
			},
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4},
			},
			expected: [][]float64{
				[]float64{0, 1},
				[]float64{4, 5},
			},
		},
		{ // Rows equal pair waise
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1, 1, 1},
			},
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4, -1, -2},
			},
			expected: [][]float64{
				[]float64{0, 1},
				[]float64{4, 5, 0, -1},
			},
		},
		{ // Src rows shorter
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1, 1, 1},
			},
			src: [][]float64{
				[]float64{1},
				[]float64{3, 4},
			},
			expected: [][]float64{
				[]float64{0, -1},
				[]float64{4, 5, 1, 1},
			},
		},
		{ // Src has less rows
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1, 1, 1},
			},
			src: [][]float64{
				[]float64{1},
			},
			expected: [][]float64{
				[]float64{0, -1},
				[]float64{1, 1, 1, 1},
			},
		},
	}

	for _, example := range examples {
		mat.SumMatrix(example.dst, example.src)
		assert.Equal(t, example.expected, example.dst)
	}
}

func TestSumMatrixPanic(t *testing.T) {
	examples := []matrixExample{
		{ // Src rows longer
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1},
			},
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4, 9999},
			},
		},
		{ // Src has more rows
			dst: [][]float64{
				[]float64{-1, -1},
				[]float64{1, 1},
			},
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4},
				[]float64{99, 99},
			},
		},
		{ // Dst not initialized
			dst: nil,
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4},
			},
		},
		{ // Dst empty
			dst: [][]float64{},
			src: [][]float64{
				[]float64{1, 2},
				[]float64{3, 4},
			},
		},
	}

	for _, example := range examples {
		assert.Panics(t, func() {
			mat.SumMatrix(example.dst, example.src)
		})
	}
}
