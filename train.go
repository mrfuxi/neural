package neural

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mrfuxi/neural/mat"
)

type WeightUpdates struct {
	Biases  [][]float64
	Weights [][][]float64
}

func NewWeightUpdates(network Evaluator) *WeightUpdates {
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

	return &WeightUpdates{
		Biases:  deltaBias,
		Weights: deltaWeights,
	}
}

func (w *WeightUpdates) Zero() {
	mat.ZeroMatrix(w.Biases)
	mat.ZeroVectorOfMatrixes(w.Weights)
}

type Trainer interface {
	StartProcessing(network Evaluator, samples <-chan TrainExample, weightPlaceholders <-chan *WeightUpdates, weightUpdates chan<- *WeightUpdates)
}

type batchRange struct {
	from, to int
}

func Train(network Evaluator, trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64, trainers ...Trainer) {
	samples := make(chan TrainExample, miniBatchSize)
	weightUpdatePlaceholders := make(chan *WeightUpdates, miniBatchSize)
	weightUpdates := make(chan *WeightUpdates, miniBatchSize)
	batchRanges := getBatchRanges(len(trainExamples), miniBatchSize)

	layers := network.Layers()

	for _, trainer := range trainers {
		go trainer.StartProcessing(network, samples, weightUpdatePlaceholders, weightUpdates)
	}

	for i := 0; i < miniBatchSize; i++ {
		weightUpdatePlaceholders <- NewWeightUpdates(network)
	}

	sumWeights := NewWeightUpdates(network)

	for epoch := 1; epoch <= epochs; epoch++ {
		shuffleTrainExamples(trainExamples)

		for b, batch := range batchRanges {
			t0 := time.Now()
			batchSize := batch.to - batch.from
			for _, sample := range trainExamples[batch.from:batch.to] {
				samples <- sample
			}

			sumWeights.Zero()
			for step := 0; step < batchSize; step++ {
				weightUpdate := <-weightUpdates
				for l := range layers {
					mat.SumMatrix(sumWeights.Weights[l], weightUpdate.Weights[l])
					mat.SumVector(sumWeights.Biases[l], weightUpdate.Biases[l])
				}
				weightUpdatePlaceholders <- weightUpdate
			}

			rate := learningRate / float64(batchSize)
			for l, layer := range layers {
				mat.MulVectorByScalar(sumWeights.Biases[l], rate)
				mat.MulMatrixByScalar(sumWeights.Weights[l], rate)
				layer.UpdateWeights(sumWeights.Weights[l], sumWeights.Biases[l])
			}

			dt := time.Since(t0)
			if b%100 == 0 {
				fmt.Printf("%v/%v %v/%v    %v     \r", epoch, epochs, b, len(batchRanges), dt)
			}
		}
	}
	close(samples)
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

type BackwardPropagationTrainer struct {
	network Evaluator
	layers  []Layer

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
}

func (b *BackwardPropagationTrainer) StartProcessing(network Evaluator, samples <-chan TrainExample, weightPlaceholders <-chan *WeightUpdates, weightUpdates chan<- *WeightUpdates) {
	b.network = network
	b.layers = b.network.Layers()

	layersCount := len(b.layers)
	b.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	b.potentialsPerLayer = make([][]float64, layersCount, layersCount)

	for l, layer := range b.layers {
		_, weightsCol, biasesCol := layer.Shapes()
		if l == 0 {
			b.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}

		b.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		b.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
	}

	for sample := range samples {
		weights := <-weightPlaceholders
		b.backwardPropagation(sample, weights)
		weightUpdates <- weights
	}
}

func (b *BackwardPropagationTrainer) backwardPropagation(sample TrainExample, weightPlaceholders *WeightUpdates) {
	layersCount := len(b.layers)

	copy(b.acticationPerLayer[0], sample.Input)
	for l, layer := range b.layers {
		layer.Forward(b.potentialsPerLayer[l], b.acticationPerLayer[l])
		b.network.Activate(b.acticationPerLayer[l+1], b.potentialsPerLayer[l], true)
	}

	errors := mat.SubVectorElementWise(b.acticationPerLayer[len(b.acticationPerLayer)-1], sample.Output)
	spOut := b.network.Activate(nil, b.potentialsPerLayer[len(b.potentialsPerLayer)-1], false)
	delta := mat.MulVectorElementWise(weightPlaceholders.Biases[layersCount-1], spOut, errors)

	mat.MulTransposeVector(weightPlaceholders.Weights[layersCount-1], delta, b.acticationPerLayer[len(b.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		sp := b.network.Activate(nil, b.potentialsPerLayer[len(b.potentialsPerLayer)-l], false)

		delta = mat.MulVectorElementWise(weightPlaceholders.Biases[layersCount-l], b.layers[layersCount-l+1].Backward(delta), sp)
		mat.MulTransposeVector(weightPlaceholders.Weights[layersCount-l], delta, b.acticationPerLayer[len(b.acticationPerLayer)-l-1])
	}
}
