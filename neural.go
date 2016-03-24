package neural

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mrfuxi/neural/mat"
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
	Train(trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64)
	TrainNew(trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64)
}

type network struct {
	activator Activator
	layers    []Layer
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

func (n *network) TrainNew(trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64) {
	trainers := make([]Trainer, 4, 4)
	for i := range trainers {
		trainers[i] = &BackwardPropagationTrainer{}
	}

	Train(n, trainExamples, epochs, miniBatchSize, learningRate, trainers...)
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

	for epoch := 1; epoch <= epochs; epoch++ {
		// Shuffle training data
		for i := range trainExamples {
			j := rand.Intn(i + 1)
			trainExamples[i], trainExamples[j] = trainExamples[j], trainExamples[i]
		}

		for b, batch := range batchRanges {
			t0 := time.Now()
			n.updateMiniBatch(trainExamples[batch.from:batch.to], learningRate)
			dt := time.Since(t0)
			if b%100 == 0 {
				fmt.Printf("%v/%v %v/%v    %v     \r", epoch, epochs, b, batches, dt)
			}
		}

	}
}

type cn struct {
	biases  [][]float64
	weights [][][]float64
}

func (n *network) updateMiniBatch(miniBatch []TrainExample, learningRate float64) {
	layersCount := len(n.layers)
	samples := len(miniBatch)

	sumDeltaBias := make([][]float64, layersCount, layersCount)
	sumDeltaWeights := make([][][]float64, layersCount, layersCount)
	buff := make(chan cn, 1000)

	for _, sample := range miniBatch {
		go func(sample TrainExample) {
			deltaWeights, deltaBias := n.backPropagation(sample)
			buff <- cn{biases: deltaBias, weights: deltaWeights}
		}(sample)
	}

	idx := 0
	for c := range buff {
		for l := range n.layers {
			if sumDeltaWeights[l] == nil {
				sumDeltaWeights[l] = mat.CopyOfMatrix(c.weights[l])
			} else {
				mat.SumMatrix(sumDeltaWeights[l], c.weights[l])
			}

			if sumDeltaBias[l] == nil {
				sumDeltaBias[l] = mat.CopyOfVector(c.biases[l])
			} else {
				mat.SumVector(sumDeltaBias[l], c.biases[l])
			}
		}

		idx++
		if idx == samples {
			close(buff)
		}
	}

	rate := learningRate / float64(samples)
	for l, layer := range n.layers {
		mat.MulVectorByScalar(sumDeltaBias[l], rate)
		mat.MulMatrixByScalar(sumDeltaWeights[l], rate)
		layer.UpdateWeights(sumDeltaWeights[l], sumDeltaBias[l])
	}
}

func (n *network) backPropagation(sample TrainExample) (deltaWeights [][][]float64, deltaBias [][]float64) {
	layersCount := len(n.layers)

	acticationPerLayer := [][]float64{}
	potentialsPerLayer := [][]float64{}

	deltaBias = make([][]float64, layersCount, layersCount)
	deltaWeights = make([][][]float64, layersCount, layersCount)

	input := sample.Input
	acticationPerLayer = append(acticationPerLayer, input)
	for _, layer := range n.layers {
		potentials := layer.Forward(nil, input)
		input = n.Activate(nil, potentials, true)
		acticationPerLayer = append(acticationPerLayer, input)
		potentialsPerLayer = append(potentialsPerLayer, potentials)
	}

	errors := mat.SubVectorElementWise(acticationPerLayer[len(acticationPerLayer)-1], sample.Output)
	spOut := n.Activate(nil, potentialsPerLayer[len(potentialsPerLayer)-1], false)
	delta := mat.MulVectorElementWise(spOut, errors)
	deltaBias[layersCount-1] = mat.CopyOfVector(delta)
	deltaWeights[layersCount-1] = mat.MulTransposeVector(deltaWeights[layersCount-1], delta, acticationPerLayer[len(acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		sp := n.Activate(nil, potentialsPerLayer[len(potentialsPerLayer)-l], false)
		delta = mat.MulVectorElementWise(n.layers[layersCount-l+1].Backward(delta), sp)
		deltaBias[layersCount-l] = mat.CopyOfVector(delta) // full copy can be avoided?
		deltaWeights[layersCount-l] = mat.MulTransposeVector(deltaWeights[layersCount-l], delta, acticationPerLayer[len(acticationPerLayer)-l-1])
	}
	return
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

// NewNeuralNetwork initializes empty neural network
func NewNeuralNetwork(activator Activator, layers ...Layer) Evaluator {
	return &network{
		activator: activator,
		layers:    layers,
	}
}
