package neural_test

import (
	"math/rand"
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func init() {
	// Predictable randomization
	rand.Seed(2)
}

func TestFeedForwardCorrentOutputSize(t *testing.T) {
	input := []float64{1, 1}
	activator := neural.NewLinearActivator(1) // identity activator
	expectedOutput := []float64{12.1, 23.2, 34.3}

	layer := neural.NewSimpleLayer(2, 3)
	layer.SetWeights(
		[][]float64{
			{0.1, 2},
			{0.2, 3},
			{0.3, 4},
		},
		[]float64{10, 20, 30},
	)

	nn := neural.NewNeuralNetwork(activator, layer)
	output := nn.Evaluate(input)

	assert.Len(t, output, 3)
	assert.Equal(t, expectedOutput, output)
}

func TestBinaryAND(t *testing.T) {
	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	}

	layer := neural.NewSimpleLayer(2, 1)
	layer.SetWeights(
		[][]float64{{1, 1}},
		[]float64{-2},
	)

	activator := neural.NewStepFunction()
	nn := neural.NewNeuralNetwork(activator, layer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.Equal(t, example.Output, output)
	}
}

func TestLearnAND(t *testing.T) {
	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	}

	outLayer := neural.NewSimpleLayer(2, 1)
	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, outLayer)

	neural.Train(nn, testMatrix, 1000, 2, 3, neural.NewBackwardPropagationTrainer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.1)
	}
}

func TestLearnOR(t *testing.T) {
	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{1}},
	}

	outLayer := neural.NewSimpleLayer(2, 1)
	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, outLayer)

	neural.Train(nn, testMatrix, 1000, 2, 3, neural.NewBackwardPropagationTrainer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.1)
	}
}

func TestUseXORStep(t *testing.T) {
	weights0 := [][]float64{{-1, 1}, {1, -1}}
	biases0 := []float64{-1, -1}
	weights1 := [][]float64{{1, 1}}
	biases1 := []float64{-0.1}

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
	}

	hiddenLayer := neural.NewSimpleLayer(2, 2)
	hiddenLayer.SetWeights(weights0, biases0)

	outLayer := neural.NewSimpleLayer(2, 1)
	outLayer.SetWeights(weights1, biases1)

	activator := neural.NewStepFunction()
	nn := neural.NewNeuralNetwork(activator, hiddenLayer, outLayer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.4999)
	}
}

func TestUseXORSigmoid(t *testing.T) {
	weights0 := [][]float64{{2.75, 2.75}, {5, 5}}
	biases0 := []float64{-4, -2}
	weights1 := [][]float64{{-6, 6}}
	biases1 := []float64{-2.5}

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
	}

	hiddenLayer := neural.NewSimpleLayer(2, 2)
	hiddenLayer.SetWeights(weights0, biases0)

	outLayer := neural.NewSimpleLayer(2, 1)
	outLayer.SetWeights(weights1, biases1)

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, hiddenLayer, outLayer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.4999)
	}
}

func TestLearnXOR(t *testing.T) {
	rand.Seed(2)

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
	}

	hiddenLayer1 := neural.NewSimpleLayer(2, 2)
	outLayer := neural.NewSimpleLayer(2, 1)

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, hiddenLayer1, outLayer)

	neural.Train(nn, testMatrix, 1000, 4, 3, neural.NewBackwardPropagationTrainer)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.2)
	}
}
