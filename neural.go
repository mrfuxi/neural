package neural

import (
	"math/rand"
	"time"
)

func init() {
	rand.Seed(time.Now().UnixNano())
}

// TrainExample represents input-output pair of signals to train on or verify the training
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

// NewNeuralNetwork initializes neural network based using activator interface, structure of neurons (counts) and layer factories
func NewNeuralNetwork(activator Activator, neurons []int, layersFactories ...LayerFactory) Evaluator {
	if len(neurons)-1 != len(layersFactories) {
		panic("Neuron counts does not match layers count")
	}

	layers := make([]Layer, len(layersFactories), len(layersFactories))
	for i, factory := range layersFactories {
		layers[i] = factory(neurons[i], neurons[i+1])
	}

	return &network{
		activator: activator,
		layers:    layers,
	}
}

// Evaluate calculates network answer for given input signal
func (n *network) Evaluate(input []float64) []float64 {
	output := input

	for _, layer := range n.layers {
		potentials := layer.Forward(nil, output)
		output = n.Activate(nil, potentials, true)
	}

	return output
}

// Layers exposes list of layers within network. Used in training only
func (n *network) Layers() []Layer {
	return n.layers
}

// Activate calculates activations or it's derivatives (in training) for given potentials
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
