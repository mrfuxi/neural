package neural

import "math/rand"

type Layer interface {
	Forward(input []float64) []float64
	Backward(delta []float64) []float64
	// SetWeights(weights *mat64.Dense, biases *mat64.Dense)
	// UpdateWeights(weights *mat64.Dense, biases *mat64.Dense)
	SetWeights(weights [][]float64, biases []float64)
	UpdateWeights(weights [][]float64, biases []float64)
}

type simpleLayer struct {
	weights [][]float64
	biases  []float64

	// weights *mat64.Dense
	// biases  *mat64.Dense
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

	// for (i = 0; i < rows; i++)
	for r, row := range s.weights {
		// fmt.Println(row, input)
		for c, inputValue := range input {
			outMat[r] += row[c] * inputValue
		}
		// outMat[r] = vectors_dot_prod(row, input)
	}

	// for
	// inputMat := mat64.NewDense(s.inputs, 1, input)
	// outMat := mat64.NewDense(s.neurons, 1, nil)

	// outMat.Mul(s.weights, inputMat)
	// outMat.Add(outMat, s.biases)

	// return outMat.RawMatrix().Data
	return outMat
}

func (s *simpleLayer) Backward(input []float64) []float64 {
	outMat := make([]float64, s.inputs, s.inputs)

	for r, row := range s.weights {
		for c := range input {
			outMat[c] += row[c] * input[r]
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
	subMatrix(s.weights, weights)
	subVector(s.biases, biases)
}
