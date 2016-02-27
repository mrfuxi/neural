package neural

// Evaluator wraps main taks of NN, evaluate input data
type Evaluator interface {
	Evaluate(input []float64) []float64
}

type network struct {
	activator Activator
	layers    []Layer
}

func (n *network) Evaluate(input []float64) []float64 {
	output := input
	for _, layer := range n.layers {
		output = layer.Forward(output)
		for i, potential := range output {
			output[i] = n.activator.Activation(potential)
		}
	}
	return output
}

// NewNeuralNetwork initializes emtpy nerual network
func NewNeuralNetwork(activator Activator, layers ...Layer) Evaluator {
	return &network{
		activator: activator,
		layers:    layers,
	}
}
