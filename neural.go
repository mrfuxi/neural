package neural

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

type TrainExample struct {
	Input  []float64
	Output []float64
}

// Evaluator wraps main tasks of NN, evaluate input data
type Evaluator interface {
	Evaluate(input []float64) []float64
	Layers() []Layer
	Activate(dst, potentials []float64, forward bool) (output []float64)
}

type network struct {
	activator Activator
	layers    []Layer
}

// NewNeuralNetwork initializes empty neural network
func NewNeuralNetwork(activator Activator, layers ...Layer) Evaluator {
	return &network{
		activator: activator,
		layers:    layers,
	}
}

func (n *network) Evaluate(input []float64) []float64 {
	output := input

	for _, layer := range n.layers {
		potentials := layer.Forward(nil, output)
		output = n.Activate(nil, potentials, true)
	}

	return output
}

func (n *network) Layers() []Layer {
	return n.layers
}

func (n *network) Activate(dst, potentials []float64, forward bool) (output []float64) {
	if dst == nil {
		dst = make([]float64, len(potentials), len(potentials))
	}

	if forward {
		for i, potential := range potentials {
			dst[i] = n.activator.Activation(potential)
		}
	} else {
		for i, potential := range potentials {
			dst[i] = n.activator.Derivative(potential)
		}
	}
	return dst
}
