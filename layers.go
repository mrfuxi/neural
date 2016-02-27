package neural

import "github.com/gonum/matrix/mat64"

type Layer interface {
	Forward(input []float64) []float64
	UpdateWeights(weights *mat64.Dense, biases *mat64.Dense)
}

type simpleLayer struct {
	weights *mat64.Dense
	biases  *mat64.Dense
	inputs  int
	neurons int
}

func NewSimpleLayer(inputs, neurons int) Layer {
	return &simpleLayer{
		weights: mat64.NewDense(neurons, inputs, nil),
		biases:  mat64.NewDense(neurons, 1, nil),
		inputs:  inputs,
		neurons: neurons,
	}
}

func (s *simpleLayer) Forward(input []float64) []float64 {
	inputMat := mat64.NewDense(s.inputs, 1, input)
	outMat := mat64.NewDense(s.neurons, 1, nil)

	outMat.Mul(s.weights, inputMat)
	outMat.Add(outMat, s.biases)

	return outMat.RawMatrix().Data
}

func (s *simpleLayer) UpdateWeights(weights *mat64.Dense, biases *mat64.Dense) {
	s.weights.Add(s.weights, weights)
	s.biases.Add(s.biases, biases)
}
