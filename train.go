package neural

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/mrfuxi/neural/mat"
)

type Trainer interface {
	StartProcessing(network Evaluator, samples <-chan TrainExample, processed chan<- bool)
	WeightUpdates() (weights [][][]float64, biases [][]float64)
	Processed() bool
	Reset()
}

type batchRange struct {
	from, to int
}

func Train(network Evaluator, trainExamples []TrainExample, epochs int, miniBatchSize int, learningRate float64, trainers ...Trainer) {
	samples := make(chan TrainExample, miniBatchSize)
	samplesProcessed := make(chan bool, miniBatchSize)
	batchRanges := getBatchRanges(len(trainExamples), miniBatchSize)

	layers := network.Layers()

	layersCount := len(layers)

	for _, trainer := range trainers {
		go trainer.StartProcessing(network, samples, samplesProcessed)
	}

	for epoch := 1; epoch <= epochs; epoch++ {
		shuffleTrainExamples(trainExamples)

		for b, batch := range batchRanges {
			t0 := time.Now()
			batchSize := batch.to - batch.from
			for _, sample := range trainExamples[batch.from:batch.to] {
				samples <- sample
			}

			// wait for all samples to be processed
			// ToDo: Replace with "for i:=0; ..."
			for range trainExamples[batch.from:batch.to] {
				<-samplesProcessed
			}

			sumDeltaBias := make([][]float64, layersCount, layersCount)
			sumDeltaWeights := make([][][]float64, layersCount, layersCount)

			for _, trainer := range trainers {
				weights, biases := trainer.WeightUpdates()
				if !trainer.Processed() {
					continue
				}

				// ToDo: Replace with "for l:=0; ..."
				for l := range layers {
					if sumDeltaWeights[l] == nil {
						sumDeltaWeights[l] = mat.CopyOfMatrix(weights[l])
					} else {
						mat.SumMatrix(sumDeltaWeights[l], weights[l])
					}

					if sumDeltaBias[l] == nil {
						sumDeltaBias[l] = mat.CopyOfVector(biases[l])
					} else {
						mat.SumVector(sumDeltaBias[l], biases[l])
					}
				}
			}

			rate := learningRate / float64(batchSize)
			for l, layer := range layers {
				mat.MulVectorByScalar(sumDeltaBias[l], rate)
				mat.MulMatrixByScalar(sumDeltaWeights[l], rate)
				layer.UpdateWeights(sumDeltaWeights[l], sumDeltaBias[l])
			}

			for _, trainer := range trainers {
				trainer.Reset()
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

	processed    bool
	deltaBias    [][]float64
	deltaWeights [][][]float64

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
}

func (b *BackwardPropagationTrainer) StartProcessing(network Evaluator, samples <-chan TrainExample, processed chan<- bool) {
	b.network = network
	b.layers = b.network.Layers()
	b.processed = false

	layersCount := len(b.layers)
	b.deltaBias = make([][]float64, layersCount, layersCount)
	b.deltaWeights = make([][][]float64, layersCount, layersCount)
	b.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	b.potentialsPerLayer = make([][]float64, layersCount, layersCount)

	for l, layer := range b.layers {
		weightsRow, weightsCol, biasesCol := layer.Shapes()
		b.deltaBias[l] = make([]float64, biasesCol, biasesCol)
		b.deltaWeights[l] = make([][]float64, weightsRow, weightsRow)

		for r := range b.deltaWeights[l] {
			b.deltaWeights[l][r] = make([]float64, weightsCol, weightsCol)
		}

		if l == 0 {
			b.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}

		b.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		b.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
	}

	for sample := range samples {
		b.backwardPropagation(sample)
		b.processed = true
		processed <- true
	}
}

func (b *BackwardPropagationTrainer) backwardPropagation(sample TrainExample) {
	layersCount := len(b.layers)

	copy(b.acticationPerLayer[0], sample.Input)
	for l, layer := range b.layers {
		layer.Forward(b.potentialsPerLayer[l], b.acticationPerLayer[l])
		b.network.Activate(b.acticationPerLayer[l+1], b.potentialsPerLayer[l], true)
	}

	errors := mat.SubVectorElementWise(b.acticationPerLayer[len(b.acticationPerLayer)-1], sample.Output)
	spOut := b.network.Activate(nil, b.potentialsPerLayer[len(b.potentialsPerLayer)-1], false)
	delta := mat.MulVectorElementWise(spOut, errors)

	mat.SumVector(b.deltaBias[layersCount-1], delta)
	mat.MulTransposeVector(b.deltaWeights[layersCount-1], delta, b.acticationPerLayer[len(b.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		sp := b.network.Activate(nil, b.potentialsPerLayer[len(b.potentialsPerLayer)-l], false)
		delta = mat.MulVectorElementWise(b.layers[layersCount-l+1].Backward(delta), sp)

		mat.SumVector(b.deltaBias[layersCount-l], delta)
		mat.MulTransposeVector(b.deltaWeights[layersCount-l], delta, b.acticationPerLayer[len(b.acticationPerLayer)-l-1])
	}
	return
}

func (b *BackwardPropagationTrainer) WeightUpdates() (weights [][][]float64, biases [][]float64) {
	return b.deltaWeights, b.deltaBias
}

func (b *BackwardPropagationTrainer) Processed() bool {
	return b.processed
}

func (b *BackwardPropagationTrainer) Reset() {
	if !b.processed {
		return
	}
	b.processed = false
	mat.ZeroMatrix(b.deltaBias)
	mat.ZeroVectorOfMatrixes(b.deltaWeights)
}
