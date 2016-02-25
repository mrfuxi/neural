package neural

// Evaluator wraps main taks of NN, evaluate input data
type Evaluator interface {
	Evaluate(input []float64) []float64
}

type network struct {
	Layers []Layer
}

func (n *network) Evaluate(input []float64) []float64 {
	output := input
	for _, layer := range n.Layers {
		output = layer.Forward(output)
	}
	return output
}

// NewNeuralNetwork initializes emtpy nerual network
func NewNeuralNetwork(layers ...Layer) Evaluator {
	return &network{
		Layers: layers,
	}
}
