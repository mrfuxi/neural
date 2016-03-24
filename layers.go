package neural

import "github.com/mrfuxi/neural/mat"

type Layer interface {
	Forward(dst, input []float64) []float64
	Backward(delta []float64) []float64
	SetWeights(weights [][]float64, biases []float64)
	UpdateWeights(weights [][]float64, biases []float64)
	Shapes() (weightsRow, weightsCol, biasesCol int)
}

type simpleLayer struct {
	weights [][]float64
	biases  []float64

	inputs  int
	neurons int
}

func NewSimpleLayer(inputs, neurons int) Layer {
	return &simpleLayer{
		weights: mat.RandomMatrix(neurons, inputs),
		biases:  mat.RandomVector(neurons),
		inputs:  inputs,
		neurons: neurons,
	}
}

func (s *simpleLayer) Shapes() (weightsRow, weightsCol, biasesCol int) {
	return s.neurons, s.inputs, s.neurons
}

func (s *simpleLayer) Forward(dst, input []float64) []float64 {
	if dst == nil {
		dst = make([]float64, s.neurons, s.neurons)
	}
	copy(dst, s.biases)

	for r, row := range s.weights {
		for c, inputValue := range input {
			dst[r] += row[c] * inputValue
		}
	}

	return dst
}

func (s *simpleLayer) Backward(input []float64) []float64 {
	outMat := make([]float64, s.inputs, s.inputs)

	for c, inputValue := range input {
		for r, rowVal := range s.weights[c] {
			outMat[r] += rowVal * inputValue
		}
	}

	return outMat
}

func (s *simpleLayer) SetWeights(weights [][]float64, biases []float64) {
	for r, row := range weights {
		copy(s.weights[r], row)
	}
	copy(s.biases, biases)
}

func (s *simpleLayer) UpdateWeights(weights [][]float64, biases []float64) {
	mat.SubMatrix(s.weights, weights)
	mat.SubVector(s.biases, biases)
}
