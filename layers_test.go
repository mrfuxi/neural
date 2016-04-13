package neural_test

import (
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestBackward(t *testing.T) {
	layerFactory := neural.NewFullyConnectedLayer(neural.NewStepFunction())
	layer := layerFactory(2, 2)
	layer.SetWeights(
		[][]float64{{1, 2}, {3, 4}},
		[]float64{-1000000},
	)

	actualBack := layer.Backward([]float64{10, 20})
	assert.EqualValues(t, []float64{70, 100}, actualBack)
}

func TestBackwardDims(t *testing.T) {
	layerFactory := neural.NewFullyConnectedLayer(neural.NewStepFunction())
	layer := layerFactory(2, 1)
	layer.SetWeights(
		[][]float64{{0.06563701921747622, 0.15651925473279124}},
		[]float64{-1000000},
	)

	actualBack := layer.Backward([]float64{0.13998155491906017})
	assert.EqualValues(t, []float64{0.009187972010314556, 0.021909808652268586}, actualBack)
}
