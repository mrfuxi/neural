package neural_test

import (
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestBackward(t *testing.T) {
	layer := neural.NewSimpleLayer(2, 2)
	layer.SetWeights(
		[][]float64{{1, 2}, {3, 4}},
		[]float64{-1000000},
	)

	actualBack := layer.Backward([]float64{10, 20})
	assert.EqualValues(t, []float64{70, 100}, actualBack)
}
