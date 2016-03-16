package neural

import (
	"math/rand"

	"github.com/mrfuxi/neural/mat"
)

type Layer interface {
	Forward(input []float64) []float64
	Backward(delta []float64) []float64
	SetWeights(weights [][]float64, biases []float64)
	UpdateWeights(weights [][]float64, biases []float64)
}

type simpleLayer struct {
	weights [][]float64
	biases  []float64

	inputs  int
	neurons int
}

func randomMatrix(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for row := range data {
		data[row] = make([]float64, cols, cols)
		for col := range data[row] {
			data[row][col] = rand.Float64()
		}
	}

	return data
}

func NewSimpleLayer(inputs, neurons int) Layer {
	return &simpleLayer{
		weights: randomMatrix(neurons, inputs),
		biases:  randomMatrix(1, neurons)[0],
		inputs:  inputs,
		neurons: neurons,
	}
}

func (s *simpleLayer) Forward(input []float64) []float64 {
	outMat := make([]float64, s.neurons, s.neurons)
	copy(outMat, s.biases)

	for r, row := range s.weights {
		for c, inputValue := range input {
			outMat[r] += row[c] * inputValue
		}
	}

	return outMat
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
