package neural

import "github.com/mrfuxi/neural/mat"

// Layer represents a single layer in nerual network
type Layer interface {
	Forward(dst, input []float64) []float64
	Backward(delta []float64) []float64
	SetWeights(weights [][]float64, biases []float64)
	UpdateWeights(weights [][]float64, biases []float64)
	Shapes() (weightsRow, weightsCol, biasesCol int)
}

// LayerFactory build a Layer of certain type, used to build a network
type LayerFactory func(inputs, neurons int) Layer

type fullyConnectedLayer struct {
	weights [][]float64
	biases  []float64

	inputs  int
	neurons int
}

// NewFullyConnectedLayer creates new neural network layer with all neurons fully connected to previous layer.
// Here it's more accruta to say it's using all input values to calculate own outputs
func NewFullyConnectedLayer(inputs, neurons int) Layer {
	return &fullyConnectedLayer{
		weights: mat.RandomMatrix(neurons, inputs),
		biases:  mat.RandomVector(neurons),
		inputs:  inputs,
		neurons: neurons,
	}
}

func (l *fullyConnectedLayer) Shapes() (weightsRow, weightsCol, biasesCol int) {
	return l.neurons, l.inputs, l.neurons
}

func (l *fullyConnectedLayer) Forward(dst, input []float64) []float64 {
	if dst == nil {
		dst = make([]float64, l.neurons, l.neurons)
	}

	tmp := 0.0
	for r, row := range l.weights {
		tmp = 0.0
		for c, inputValue := range input {
			tmp += row[c] * inputValue
		}
		dst[r] = tmp + l.biases[r]
	}

	return dst
}

func (l *fullyConnectedLayer) Backward(input []float64) []float64 {
	outMat := make([]float64, l.inputs, l.inputs)

	for c, inputValue := range input {
		for r, rowVal := range l.weights[c] {
			outMat[r] += rowVal * inputValue
		}
	}

	return outMat
}

func (l *fullyConnectedLayer) SetWeights(weights [][]float64, biases []float64) {
	for r, row := range weights {
		copy(l.weights[r], row)
	}
	copy(l.biases, biases)
}

func (l *fullyConnectedLayer) UpdateWeights(weights [][]float64, biases []float64) {
	mat.SubMatrix(l.weights, weights)
	mat.SubVector(l.biases, biases)
}
