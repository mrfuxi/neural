package neural_test

import (
	"math/rand"
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/mrfuxi/neural/mat"
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

	nn := neural.NewNeuralNetwork([]int{2, 3}, neural.NewFullyConnectedLayer(activator))
	nn.Layers()[0].SetWeights(
		[][]float64{
			{0.1, 2},
			{0.2, 3},
			{0.3, 4},
		},
		[]float64{10, 20, 30},
	)

	output := nn.Evaluate(input)

	assert.Len(t, output, 3)
	assert.Equal(t, expectedOutput, output)
}

func TestDifferentShapes(t *testing.T) {
	activator := neural.NewSigmoidActivator()

	options := neural.TrainOptions{
		Epochs:         5,
		MiniBatchSize:  2,
		LearningRate:   3,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewQuadraticCost(),
	}

	for in := 1; in < 101; in += 20 {
		for hid := 1; hid < 101; hid += 20 {
			for out := 1; out < 101; out += 20 {
				nn := neural.NewNeuralNetwork(
					[]int{in, hid, out},
					neural.NewFullyConnectedLayer(activator),
					neural.NewFullyConnectedLayer(activator),
				)

				testMatrix := []neural.TrainExample{
					{Input: mat.RandomVector(in), Output: mat.RandomVector(out)},
					{Input: mat.RandomVector(in), Output: mat.RandomVector(out)},
					{Input: mat.RandomVector(in), Output: mat.RandomVector(out)},
					{Input: mat.RandomVector(in), Output: mat.RandomVector(out)},
					{Input: mat.RandomVector(in), Output: mat.RandomVector(out)},
				}

				// Training should not cause any errors
				neural.Train(nn, testMatrix, options)

				// Evaluation should not cause any errors
				output := nn.Evaluate(mat.RandomVector(in))

				assert.Len(t, output, out)
			}
		}
	}
}

func TestBinaryAND(t *testing.T) {
	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	}

	activator := neural.NewStepActivator()
	nn := neural.NewNeuralNetwork([]int{2, 1}, neural.NewFullyConnectedLayer(activator))
	nn.Layers()[0].SetWeights(
		[][]float64{{1, 1}},
		[]float64{-2},
	)

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

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork([]int{2, 1}, neural.NewFullyConnectedLayer(activator))

	options := neural.TrainOptions{
		Epochs:         1000,
		MiniBatchSize:  2,
		LearningRate:   3,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewQuadraticCost(),
	}
	neural.Train(nn, testMatrix, options)

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

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork([]int{2, 1}, neural.NewFullyConnectedLayer(activator))

	options := neural.TrainOptions{
		Epochs:         1000,
		MiniBatchSize:  2,
		LearningRate:   3,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewQuadraticCost(),
	}
	neural.Train(nn, testMatrix, options)

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

	activator := neural.NewStepActivator()
	nn := neural.NewNeuralNetwork([]int{2, 2, 1}, neural.NewFullyConnectedLayer(activator), neural.NewFullyConnectedLayer(activator))
	nn.Layers()[0].SetWeights(weights0, biases0)
	nn.Layers()[1].SetWeights(weights1, biases1)

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

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork([]int{2, 2, 1}, neural.NewFullyConnectedLayer(activator), neural.NewFullyConnectedLayer(activator))
	nn.Layers()[0].SetWeights(weights0, biases0)
	nn.Layers()[1].SetWeights(weights1, biases1)

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

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork([]int{2, 2, 1}, neural.NewFullyConnectedLayer(activator), neural.NewFullyConnectedLayer(activator))

	options := neural.TrainOptions{
		Epochs:         1020,
		MiniBatchSize:  4,
		LearningRate:   3,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewQuadraticCost(),
	}
	neural.Train(nn, testMatrix, options)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.2)
	}
}

func TestLearnXORCrossEntropy(t *testing.T) {
	rand.Seed(2)

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
	}

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork([]int{2, 2, 1}, neural.NewFullyConnectedLayer(activator), neural.NewFullyConnectedLayer(activator))

	options := neural.TrainOptions{
		Epochs:         260,
		MiniBatchSize:  4,
		LearningRate:   3,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewCrossEntropyCost(),
	}
	neural.Train(nn, testMatrix, options)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDelta(t, example.Output[0], output[0], 0.2)
	}
}

func TestLearnXORSoftMaxLogLikelyhood(t *testing.T) {
	rand.Seed(2)

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0, 1}},
		{[]float64{1, 1}, []float64{0, 1}},
		{[]float64{0, 1}, []float64{1, 0}},
		{[]float64{1, 0}, []float64{1, 0}},
	}

	sigmoidActivator := neural.NewSigmoidActivator()
	softMaxActivator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(
		[]int{2, 2, 2},
		neural.NewFullyConnectedLayer(sigmoidActivator),
		neural.NewFullyConnectedLayer(softMaxActivator),
	)

	options := neural.TrainOptions{
		Epochs:         173,
		MiniBatchSize:  4,
		LearningRate:   3,
		Momentum:       0,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewLogLikelihoodCost(),
	}
	neural.Train(nn, testMatrix, options)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDeltaSlice(t, example.Output, output, 0.2)
	}
}

func TestLearnXORSoftMaxLogLikelyhoodMomentum(t *testing.T) {
	rand.Seed(2)

	testMatrix := []neural.TrainExample{
		{[]float64{0, 0}, []float64{0, 1}},
		{[]float64{1, 1}, []float64{0, 1}},
		{[]float64{0, 1}, []float64{1, 0}},
		{[]float64{1, 0}, []float64{1, 0}},
	}

	sigmoidActivator := neural.NewSigmoidActivator()
	softMaxActivator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(
		[]int{2, 2, 2},
		neural.NewFullyConnectedLayer(sigmoidActivator),
		neural.NewFullyConnectedLayer(softMaxActivator),
	)

	options := neural.TrainOptions{
		Epochs:         122,
		MiniBatchSize:  4,
		LearningRate:   3,
		Momentum:       0.9,
		TrainerFactory: neural.NewBackpropagationTrainer,
		Cost:           neural.NewLogLikelihoodCost(),
	}
	neural.Train(nn, testMatrix, options)

	for _, example := range testMatrix {
		output := nn.Evaluate(example.Input)
		assert.InDeltaSlice(t, example.Output, output, 0.2)
	}
}
