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

func TestSumVectors(t *testing.T) {
	dst := []float64{-1, -1, -1, -1, -1}
	v1 := []float64{0, 1, 2, 3, 4}
	v2 := []float64{0, 10, 20, 30, 40}
	v3 := []float64{0, 100, 200, 300, 400}

	mat.SumVectors(dst, v1, v2, v3)

	assert.Equal(t, dst, []float64{0, 111, 222, 333, 444})
}

func TestSumMatrixes(t *testing.T) {
	dst := [][]float64{
		[]float64{-1, -1, -1},
		[]float64{-1, -1, -1},
	}
	v1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{4, 5, 6},
	}
	v2 := [][]float64{
		[]float64{10, 20, 30},
		[]float64{40, 50, 60},
	}
	v3 := [][]float64{
		[]float64{100, 200, 300},
		[]float64{400, 500, 600},
	}

	expected := [][]float64{
		[]float64{111, 222, 333},
		[]float64{444, 555, 666},
	}

	mat.SumMatrixes(dst, v1, v2, v3)

	assert.Equal(t, dst, expected)
}
