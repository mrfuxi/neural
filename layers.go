package neural

import (
	"encoding/gob"
	"io"
	"math"

	"github.com/mrfuxi/neural/mat"
)

// Layer represents a single layer in nerual network
type Layer interface {
	Forward(dst, input []float64)
	Backward(dst, delta []float64)
	SetWeights(weights [][]float64, biases []float64)
	UpdateWeights(weights [][]float64, biases []float64, regularization float64)
	Shapes() (weightsRow, weightsCol, biasesCol int)
	Activator() Activator
	SaverLoader
}

// LayerFactory build a Layer of certain type, used to build a network
type LayerFactory func(inputs, neurons int) Layer

type fullyConnectedLayer struct {
	activator Activator
	weights   [][]float64
	biases    []float64

	inputs  int
	neurons int
}

// NewFullyConnectedLayer creates new neural network layer with all neurons fully connected to previous layer.
// Here it's more accruta to say it's using all input values to calculate own outputs
// func NewFullyConnectedLayer(inputs, neurons int, activator Activator) Layer {
func NewFullyConnectedLayer(activator Activator) LayerFactory {
	return func(inputs, neurons int) Layer {

		weightsNorm := 1 / math.Sqrt(float64(inputs))
		weights := mat.RandomMatrix(neurons, inputs)
		mat.MulMatrixByScalar(weights, weightsNorm)

		return &fullyConnectedLayer{
			weights:   weights,
			biases:    mat.RandomVector(neurons),
			inputs:    inputs,
			neurons:   neurons,
			activator: activator,
		}
	}
}

func (l *fullyConnectedLayer) Shapes() (weightsRow, weightsCol, biasesCol int) {
	return l.neurons, l.inputs, l.neurons
}

func (l *fullyConnectedLayer) Forward(dst, input []float64) {
	tmp := 0.0
	for r, row := range l.weights {
		tmp = 0.0
		for c, inputValue := range input {
			tmp += row[c] * inputValue
		}
		dst[r] = tmp + l.biases[r]
	}
}

func (l *fullyConnectedLayer) Backward(dst, input []float64) {
	for i := range dst {
		dst[i] = 0
	}

	for c, inputValue := range input {
		for r, rowVal := range l.weights[c] {
			dst[r] += rowVal * inputValue
		}
	}
}

func (l *fullyConnectedLayer) SetWeights(weights [][]float64, biases []float64) {
	for r, row := range weights {
		copy(l.weights[r], row)
	}
	copy(l.biases, biases)
}

func (l *fullyConnectedLayer) UpdateWeights(weights [][]float64, biases []float64, regularization float64) {
	if regularization != 1 {
		mat.MulMatrixByScalar(l.weights, regularization)
	}
	mat.SumMatrix(l.weights, weights)
	mat.SumVector(l.biases, biases)
}

func (l *fullyConnectedLayer) Activator() Activator {
	return l.activator
}

func (l *fullyConnectedLayer) Save(w io.Writer) error {
	encoder := gob.NewEncoder(w)

	err := encoder.Encode(l.biases)
	if err != nil {
		return err
	}

	err = encoder.Encode(l.weights)
	if err != nil {
		return err
	}

	return nil
}

func (l *fullyConnectedLayer) Load(r io.Reader) error {
	decoder := gob.NewDecoder(r)

	if err := decoder.Decode(&l.biases); err != nil {
		return err
	}

	if err := decoder.Decode(&l.weights); err != nil {
		return err
	}
	return nil
}
