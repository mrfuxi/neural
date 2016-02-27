package neural_test

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestFeedForwardCorrentOutputSize(t *testing.T) {
	input := []float64{1, 1}
	activator := neural.NewLinearActivator(1) // identity activator
	// expectedOutput := []float64{1, 1, 1}

	layer := neural.NewSimpleLayer(2, 3)
	nn := neural.NewNeuralNetwork(activator, layer)
	output := nn.Evaluate(input)

	assert.Len(t, output, 3)
	// assert.Equal(t, expectedOutput, output)
}

func TestBinaryAND(t *testing.T) {
	testMatrix := []struct {
		input  []float64
		output []float64
	}{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	}

	layer := neural.NewSimpleLayer(2, 1)
	layer.UpdateWeights(
		mat64.NewDense(1, 2, []float64{1, 1}),
		mat64.NewDense(1, 1, []float64{-2}),
	)

	activator := neural.NewStepFunction()
	nn := neural.NewNeuralNetwork(activator, layer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.input)
		assert.Equal(t, example.output, output)
	}
}
