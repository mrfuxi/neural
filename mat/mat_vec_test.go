package mat_test

import (
	"testing"

	"github.com/mrfuxi/neural/mat"
	"github.com/stretchr/testify/assert"
)

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
