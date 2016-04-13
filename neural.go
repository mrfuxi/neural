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
}

type network struct {
	layers []Layer
}

// NewNeuralNetwork initializes neural network structure of neurons (counts) and layer factories
func NewNeuralNetwork(neurons []int, layersFactories ...LayerFactory) Evaluator {
	if len(neurons)-1 != len(layersFactories) {
		panic("Neuron counts does not match layers count")
	}

	layers := make([]Layer, len(layersFactories), len(layersFactories))
	for i, factory := range layersFactories {
		layers[i] = factory(neurons[i], neurons[i+1])
	}

	return &network{
		layers: layers,
	}
}

// Evaluate calculates network answer for given input signal
func (n *network) Evaluate(input []float64) []float64 {
	output := input

	for _, layer := range n.layers {
		potentials := layer.Forward(nil, output)
		output = make([]float64, len(potentials), len(potentials))
		layer.Activator().Activation(output, potentials)
	}

	return output
}

// Layers exposes list of layers within network. Used in training only
func (n *network) Layers() []Layer {
	return n.layers
}
