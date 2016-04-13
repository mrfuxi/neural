package neural

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mrfuxi/neural/mat"
)

// WeightUpdates is per Layer representation of how to adjust weights of the network
type WeightUpdates struct {
	Biases  [][]float64
	Weights [][][]float64
}

// Trainer implements calculations of weights adjustments (WeightUpdates) in the network
// It operates on a single training example to prepare fractional result
type Trainer interface {
	Process(sample TrainExample, weightUpdates *WeightUpdates)
}

// TrainerFactory build Trainers. Multiple trainers will be created at the beginning of the training.
type TrainerFactory func(network Evaluator) Trainer

// NewWeightUpdates creates WeightUpdates according to structure of the network (neurons in each layer)
func NewWeightUpdates(network Evaluator) WeightUpdates {
	layers := network.Layers()
	layersCount := len(layers)
	deltaBias := make([][]float64, layersCount, layersCount)
	deltaWeights := make([][][]float64, layersCount, layersCount)

	for l, layer := range layers {
		weightsRow, weightsCol, biasesCol := layer.Shapes()
		deltaBias[l] = make([]float64, biasesCol, biasesCol)
		deltaWeights[l] = make([][]float64, weightsRow, weightsRow)

		for r := range deltaWeights[l] {
			deltaWeights[l][r] = make([]float64, weightsCol, weightsCol)
		}
	}

	return WeightUpdates{
		Biases:  deltaBias,
		Weights: deltaWeights,
	}
}

// Zero sets all weights values to 0
func (w *WeightUpdates) Zero() {
	mat.ZeroMatrix(w.Biases)
	mat.ZeroVectorOfMatrixes(w.Weights)
}

type batchRange struct {
	from, to int
}

// Train executes training algorithm using provided Trainers (build with TrainerFactory)
// Training happens in randomized batches where samples are processed concurrently
func Train(network Evaluator, trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64, trainerFactory TrainerFactory) {
	batchRanges := getBatchRanges(len(trainExamples), miniBatchSize)
	ready := make(chan int, miniBatchSize)

	layers := network.Layers()

	trainers := make([]Trainer, miniBatchSize, miniBatchSize)
	for i := range trainers {
		trainers[i] = trainerFactory(network)
	}

	weightUpdates := make([]WeightUpdates, miniBatchSize, miniBatchSize)
	for i := range weightUpdates {
		weightUpdates[i] = NewWeightUpdates(network)
	}

	sumWeights := NewWeightUpdates(network)

	for epoch := 1; epoch <= epochs; epoch++ {
		shuffleTrainExamples(trainExamples)

		for b, batch := range batchRanges {
			t0 := time.Now()
			samples := trainExamples[batch.from:batch.to]
			for i := range samples {
				go func(i int) {
					trainers[i].Process(samples[i], &weightUpdates[i])
					ready <- i
				}(i)
			}

			sumWeights.Zero()
			batchSize := batch.to - batch.from
			processed := 0
			for i := range ready {
				weightUpdate := weightUpdates[i]
				for l := range layers {
					mat.SumMatrix(sumWeights.Weights[l], weightUpdate.Weights[l])
					mat.SumVector(sumWeights.Biases[l], weightUpdate.Biases[l])
				}
				processed++
				if processed == batchSize {
					break
				}
			}

			rate := learningRate / float64(batchSize)
			for l, layer := range layers {
				mat.MulVectorByScalar(sumWeights.Biases[l], rate)
				mat.MulMatrixByScalar(sumWeights.Weights[l], rate)
				layer.UpdateWeights(sumWeights.Weights[l], sumWeights.Biases[l])
			}

			if b%1000 == 0 {
				dt := time.Since(t0)
				fmt.Printf("%v/%v %v/%v    %v     \r", epoch, epochs, b, len(batchRanges), dt)
			}
		}
	}
}

func shuffleTrainExamples(trainExamples []TrainExample) {
	for i := range trainExamples {
		j := rand.Intn(i + 1)
		trainExamples[i], trainExamples[j] = trainExamples[j], trainExamples[i]
	}
}

func getBatchRanges(samples, miniBatchSize int) []batchRange {
	batches := samples / miniBatchSize
	if samples%miniBatchSize != 0 {
		batches++
	}

	batchRanges := make([]batchRange, batches, batches)
	for b := range batchRanges {
		min := b * miniBatchSize
		max := min + miniBatchSize
		if max > samples {
			max = samples
		}
		batchRanges[b] = batchRange{min, max}
	}

	return batchRanges
}

type backwardPropagationTrainer struct {
	network Evaluator
	layers  []Layer

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
}

// NewBackwardPropagationTrainer builds new trainer that uses backward propagation algorithm
func NewBackwardPropagationTrainer(network Evaluator) Trainer {
	trainer := backwardPropagationTrainer{
		network: network,
		layers:  network.Layers(),
	}

	layersCount := len(trainer.layers)
	trainer.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	trainer.potentialsPerLayer = make([][]float64, layersCount, layersCount)

	for l, layer := range trainer.layers {
		_, weightsCol, biasesCol := layer.Shapes()
		if l == 0 {
			trainer.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}

		trainer.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		trainer.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
	}
	return &trainer
}

// Process executes backward propagation algorithm to get weight updates
func (b *backwardPropagationTrainer) Process(sample TrainExample, weightUpdates *WeightUpdates) {
	layersCount := len(b.layers)

	copy(b.acticationPerLayer[0], sample.Input)
	for l, layer := range b.layers {
		layer.Forward(b.potentialsPerLayer[l], b.acticationPerLayer[l])
		layer.Activator().Activation(b.acticationPerLayer[l+1], b.potentialsPerLayer[l])
	}

	spOut := make([]float64, len(sample.Output), len(sample.Output))
	errors := mat.SubVectorElementWise(b.acticationPerLayer[len(b.acticationPerLayer)-1], sample.Output)
	b.network.Layers()[layersCount-1].Activator().Derivative(spOut, b.potentialsPerLayer[len(b.potentialsPerLayer)-1])
	delta := mat.MulVectorElementWise(weightUpdates.Biases[layersCount-1], spOut, errors)

	mat.MulTransposeVector(weightUpdates.Weights[layersCount-1], delta, b.acticationPerLayer[len(b.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		lNo := layersCount - l
		potentials := b.potentialsPerLayer[len(b.potentialsPerLayer)-l]
		sp := make([]float64, len(potentials), len(potentials))
		b.network.Layers()[lNo].Activator().Derivative(sp, potentials)

		delta = mat.MulVectorElementWise(weightUpdates.Biases[lNo], b.layers[lNo+1].Backward(delta), sp)
		mat.MulTransposeVector(weightUpdates.Weights[lNo], delta, b.acticationPerLayer[len(b.acticationPerLayer)-l-1])
	}
}
