package neural

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

type TrainExample struct {
	Input  []float64
	Output []float64
}

// Evaluator wraps main tasks of NN, evaluate input data
type Evaluator interface {
	Evaluate(input []float64) []float64
	Train(trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64)
}

type network struct {
	activator Activator
	layers    []Layer
}

func (n *network) Evaluate(input []float64) []float64 {
	output := input

	for _, layer := range n.layers {
		potentials := layer.Forward(output)
		output = n.Activate(potentials, true)
	}

	return output
}

func (n *network) Train(trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64) {
	type Range struct {
		from, to int
	}

	samples := len(trainExamples)
	batches := samples / miniBatchSize
	if len(trainExamples)%miniBatchSize != 0 {
		batches++
	}

	batchRanges := make([]Range, batches, batches)
	for b := range batchRanges {
		min := b * miniBatchSize
		max := min + miniBatchSize
		if max > samples {
			max = samples
		}
		batchRanges[b] = Range{min, max}
	}

	for epoch := 0; epoch <= epochs; epoch++ {
		// Shuffle training data
		for i := range trainExamples {
			j := rand.Intn(i + 1)
			trainExamples[i], trainExamples[j] = trainExamples[j], trainExamples[i]
		}

		for _, batch := range batchRanges {
			n.updateMiniBatch(trainExamples[batch.from:batch.to], learningRate)
		}
	}
}

func (n *network) updateMiniBatch(miniBatch []TrainExample, learningRate float64) {
	layersCount := len(n.layers)
	samples := len(miniBatch)

	sumDeltaBias := make([]*mat64.Dense, layersCount, layersCount)
	sumDeltaWeights := make([]*mat64.Dense, layersCount, layersCount)

	for _, sample := range miniBatch {
		deltaWeights, deltaBias := n.backPropagation(sample)

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

	rate := learningRate / float64(samples)
	for l, layer := range n.layers {
		sumDeltaWeights[l].Scale(rate, sumDeltaWeights[l])
		sumDeltaBias[l].Scale(rate, sumDeltaBias[l])
		layer.UpdateWeights(sumDeltaWeights[l], sumDeltaBias[l])
	}
}

func (n *network) backPropagation(sample TrainExample) (deltaWeights []*mat64.Dense, deltaBias []*mat64.Dense) {
	layersCount := len(n.layers)

	acticationPerLayer := [][]float64{}
	potentialsPerLayer := [][]float64{}
	deltaBias = make([]*mat64.Dense, layersCount, layersCount)
	deltaWeights = make([]*mat64.Dense, layersCount, layersCount)

	input := sample.Input
	acticationPerLayer = append(acticationPerLayer, input)
	for _, layer := range n.layers {
		potentials := layer.Forward(input)
		input = n.Activate(potentials, true)
		acticationPerLayer = append(acticationPerLayer, input)
		potentialsPerLayer = append(potentialsPerLayer, potentials)
	}

	errors := n.Diff(acticationPerLayer[len(acticationPerLayer)-1], sample.Output)
	delta := n.Delta(potentialsPerLayer[len(potentialsPerLayer)-1], errors)
	deltaBias[layersCount-1] = mat64.NewDense(len(delta), 1, delta)
	deltaWeights[layersCount-1] = n.MulTranspose(delta, acticationPerLayer[len(acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		sp := n.Activate(potentialsPerLayer[len(potentialsPerLayer)-l], false)
		delta = n.Mul(n.layers[layersCount-l+1].Backward(delta), sp)
		deltaBias[layersCount-l] = mat64.NewDense(len(delta), 1, delta)
		deltaWeights[layersCount-l] = n.MulTranspose(delta, acticationPerLayer[len(acticationPerLayer)-l-1])
	}
	return
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
