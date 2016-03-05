package neural

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

type TrainExample struct {
	Input  []float64
	Output []float64
}

// Evaluator wraps main taks of NN, evaluate input data
type Evaluator interface {
	Evaluate(input []float64) []float64
	Train(trainExamples []TrainExample)
}

type network struct {
	activator Activator
	layers    []Layer
}

func (n *network) Evaluate(input []float64) []float64 {
	output := input
	// fmt.Println("input:", input)

	for _, layer := range n.layers {
		potentials := layer.Forward(output)
		// fmt.Printf("%v p: %v\n", i, potentials)
		output = n.Activate(potentials, true)
		// fmt.Printf("%v a: %v\n", i, potentials)
	}
	// fmt.Println("output:", output)

	return output
}

func (n *network) Train(trainExamples []TrainExample) {
	layersCount := len(n.layers)

	sumDeltaBias := make([]*mat64.Dense, layersCount, layersCount)
	sumDeltaWeights := make([]*mat64.Dense, layersCount, layersCount)

	for _, sample := range trainExamples {
		acticationPerLayer := make([][]float64, 0)
		potentialsPerLayer := make([][]float64, 0)
		deltaBias := make([]*mat64.Dense, layersCount, layersCount)
		deltaWeights := make([]*mat64.Dense, layersCount, layersCount)

		input := sample.Input
		acticationPerLayer = append(acticationPerLayer, input)
		for _, layer := range n.layers {
			potentials := layer.Forward(input)
			input = n.Activate(potentials, true)
			acticationPerLayer = append(acticationPerLayer, input)
			potentialsPerLayer = append(potentialsPerLayer, potentials)
		}
		// fmt.Println("acticationPerLayer:", acticationPerLayer)
		// fmt.Println("potentialsPerLayer:", potentialsPerLayer)

		// fmt.Println("act:", acticationPerLayer[len(acticationPerLayer)-1])
		// fmt.Println("expected:", sample.Output)
		// fmt.Println("out layer:", layersCount-1)

		errors := n.Diff(acticationPerLayer[len(acticationPerLayer)-1], sample.Output)
		delta := n.Delta(potentialsPerLayer[len(potentialsPerLayer)-1], errors)
		deltaBias[layersCount-1] = mat64.NewDense(len(delta), 1, delta)
		deltaWeights[layersCount-1] = n.MulTranspose(delta, acticationPerLayer[len(acticationPerLayer)-2])
		// fmt.Println("potentials:", potentialsPerLayer[len(potentialsPerLayer)-1])
		// fmt.Println("potentials derivative:", n.Activate(potentialsPerLayer[len(potentialsPerLayer)-1], false))
		// fmt.Println("potentials act:", n.Activate(potentialsPerLayer[len(potentialsPerLayer)-1], true))
		// fmt.Println("errors:", errors)
		// fmt.Println("delta:", delta)
		// fmt.Println("a:", acticationPerLayer[len(acticationPerLayer)-2])
		// fmt.Println("dw:", deltaWeights[layersCount-1].RawMatrix().Data)

		for l := 2; l <= layersCount; l++ {
			// fmt.Println("layer:", layersCount-l)
			sp := n.Activate(potentialsPerLayer[len(potentialsPerLayer)-l], false)
			delta = n.Mul(n.layers[layersCount-l+1].Backward(delta), sp)
			// fmt.Println("delta:", delta)
			// delta = n.Delta(...)
			deltaBias[layersCount-l] = mat64.NewDense(len(delta), 1, delta)
			deltaWeights[layersCount-l] = n.MulTranspose(delta, acticationPerLayer[len(acticationPerLayer)-l-0])
			// fmt.Println("dw:", deltaWeights[layersCount-l])
		}

		for l := range n.layers {
			if sumDeltaBias[l] == nil {
				sumDeltaBias[l] = mat64.DenseCopyOf(deltaBias[l])
			} else {
				sumDeltaBias[l].Add(sumDeltaBias[l], deltaBias[l])
			}

			if sumDeltaWeights[l] == nil {
				sumDeltaWeights[l] = mat64.DenseCopyOf(deltaWeights[l])
			} else {
				sumDeltaWeights[l].Add(sumDeltaWeights[l], deltaWeights[l])
			}
		}
	}

	// fmt.Println("Update by:")
	for l, layer := range n.layers {
		sumDeltaWeights[l].Scale(0.5, sumDeltaWeights[l])
		sumDeltaBias[l].Scale(0.5, sumDeltaBias[l])

		// fw := mat64.Formatted(sumDeltaWeights[l], mat64.Prefix("    "))
		// fb := mat64.Formatted(sumDeltaBias[l], mat64.Prefix("    "))
		// fmt.Printf("w = %v\n\n", fw)
		// fmt.Printf("b = %v\n\n", fb)

		layer.UpdateWeights(sumDeltaWeights[l], sumDeltaBias[l])
	}
	// fmt.Println("Done")
}

func (n *network) Activate(potentials []float64, forward bool) (output []float64) {
	output = make([]float64, len(potentials), len(potentials))

	if forward {
		for i, potential := range potentials {
			output[i] = n.activator.Activation(potential)
		}
	} else {
		for i, potential := range potentials {
			output[i] = n.activator.Derivative(potential)
		}
	}
	return
}

func (n *network) MulTranspose(a, b []float64) (mul *mat64.Dense) {
	// if len(a) != len(b) {
	// 	errMsg := fmt.Sprintf("Incompatible sizes. %v vs %v", len(a), len(b))
	// 	panic(errMsg)
	// }

	// mul = make([]float64, len(a), len(a))
	// for i := range mul {
	// sp := mul[i] )
	// 	= a[i] * b[i]  matA :=
	// }
	matA := mat64.NewDense(len(a), 1, a)
	matB := mat64.NewDense(len(b), 1, b)
	mul = mat64.NewDense(len(a), len(b), nil)

	mul.Mul(matA, matB.T())
	return
}

func (n *network) Mul(a, b []float64) (mul []float64) {
	if len(a) != len(b) {
		errMsg := fmt.Sprintf("Incompatible sizes. %v vs %v", len(a), len(b))
		panic(errMsg)
	}

	mul = make([]float64, len(a), len(a))
	for i := range mul {
		mul[i] = a[i] * b[i]
	}
	return mul
}

func (n *network) Diff(a, b []float64) (diff []float64) {
	if len(a) != len(b) {
		errMsg := fmt.Sprintf("Incompatible sizes. %v vs %v", len(a), len(b))
		panic(errMsg)
	}

	diff = make([]float64, len(a), len(a))
	for i := range diff {
		diff[i] = a[i] - b[i]
	}
	return
}

func (n *network) Delta(potentials, errors []float64) (delta []float64) {
	if len(potentials) != len(errors) {
		errMsg := fmt.Sprintf("Incompatible sizes. %v vs %v", len(potentials), len(errors))
		panic(errMsg)
	}

	delta = make([]float64, len(potentials), len(potentials))
	for i := range potentials {
		delta[i] = errors[i] * n.activator.Derivative(potentials[i])
	}
	return
}

// NewNeuralNetwork initializes empty neural network
func NewNeuralNetwork(activator Activator, layers ...Layer) Evaluator {
	return &network{
		activator: activator,
		layers:    layers,
	}
}
