package neural

import (
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

type Layer interface {
	Forward(input []float64) []float64
	Backward(delta []float64) []float64
	SetWeights(weights *mat64.Dense, biases *mat64.Dense)
	UpdateWeights(weights *mat64.Dense, biases *mat64.Dense)
}

type simpleLayer struct {
	weights *mat64.Dense
	biases  *mat64.Dense
	inputs  int
	neurons int
}

func randomMatrix(rows, cols int) *mat64.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()
	}

	return mat64.NewDense(rows, cols, data)
}

func NewSimpleLayer(inputs, neurons int) Layer {
	return &simpleLayer{
		weights: randomMatrix(neurons, inputs),
		biases:  randomMatrix(neurons, 1),
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

func (s *simpleLayer) Backward(input []float64) []float64 {
	inputMat := mat64.NewDense(len(input), 1, input)
	outMat := mat64.NewDense(s.inputs, len(input), nil)

	outMat.Mul(s.weights.T(), inputMat)

	return outMat.RawMatrix().Data
}

func (s *simpleLayer) SetWeights(weights *mat64.Dense, biases *mat64.Dense) {
	s.weights.Clone(weights)
	s.biases.Clone(biases)
}

func (s *simpleLayer) UpdateWeights(weights *mat64.Dense, biases *mat64.Dense) {
	s.weights.Sub(s.weights, weights)
	s.biases.Sub(s.biases, biases)
}
