package neural_test

import (
	"bytes"
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestSaveLoad(t *testing.T) {
	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
	}

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, []int{2, 2, 1}, neural.NewFullyConnectedLayer, neural.NewFullyConnectedLayer)
	nn.Layers()[0].SetWeights([][]float64{{2.75, 2.75}, {5, 5}}, []float64{-4, -2})
	nn.Layers()[1].SetWeights([][]float64{{-6, 6}}, []float64{-2.5})

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.4999)
	}

	// Save
	buffer := new(bytes.Buffer)
	neural.Save(nn, buffer)

	// New empty NN
	newNn := neural.NewNeuralNetwork(activator, []int{2, 2, 1}, neural.NewFullyConnectedLayer, neural.NewFullyConnectedLayer)
	newNn.Layers()[0].SetWeights([][]float64{{0, 0}, {0, 0}}, []float64{0, 0})
	newNn.Layers()[1].SetWeights([][]float64{{0, 0}}, []float64{0})

	// NN responses to 0.5 for all cases - wrong
	for _, example := range testMatrix {
		output := newNn.Evaluate(example.Input)
		assert.InDelta(t, 0.5, output[0], 0.0001)
	}

	// Load
	neural.Load(newNn, buffer)
	for _, example := range testMatrix {
		output := newNn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.4999)
	}
}
