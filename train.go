package neural

import (
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
type TrainerFactory func(network Evaluator, cost Cost) Trainer

// EpocheCallback gets called at the end of every epoche with information about the state of training
type EpocheCallback func(epoche int, dt time.Duration)

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

// TrainOptions define different switches used to train a network
type TrainOptions struct {
	Epochs         int
	MiniBatchSize  int
	LearningRate   float64
	Cost           Cost
	TrainerFactory TrainerFactory
	EpocheCallback EpocheCallback
}

// Train executes training algorithm using provided Trainers (build with TrainerFactory)
// Training happens in randomized batches where samples are processed concurrently
func Train(network Evaluator, trainExamples []TrainExample, options TrainOptions) {
	batchRanges := getBatchRanges(len(trainExamples), options.MiniBatchSize)
	ready := make(chan int, options.MiniBatchSize)

	layers := network.Layers()

	trainers := make([]Trainer, options.MiniBatchSize, options.MiniBatchSize)
	for i := range trainers {
		trainers[i] = options.TrainerFactory(network, options.Cost)
	}

	weightUpdates := make([]WeightUpdates, options.MiniBatchSize, options.MiniBatchSize)
	for i := range weightUpdates {
		weightUpdates[i] = NewWeightUpdates(network)
	}

	sumWeights := NewWeightUpdates(network)

	for epoch := 1; epoch <= options.Epochs; epoch++ {
		shuffleTrainExamples(trainExamples)
		t0 := time.Now()

		for _, batch := range batchRanges {
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

			rate := options.LearningRate / float64(batchSize)
			for l, layer := range layers {
				mat.MulVectorByScalar(sumWeights.Biases[l], rate)
				mat.MulMatrixByScalar(sumWeights.Weights[l], rate)
				layer.UpdateWeights(sumWeights.Weights[l], sumWeights.Biases[l])
			}
		}

		if options.EpocheCallback != nil {
			dt := time.Since(t0)
			options.EpocheCallback(epoch, dt)
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
	cost    Cost

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
	outError           []float64
	sp                 [][]float64
	backward           [][]float64
}

// NewBackwardPropagationTrainer builds new trainer that uses backward propagation algorithm
func NewBackwardPropagationTrainer(network Evaluator, cost Cost) Trainer {
	trainer := backwardPropagationTrainer{
		network: network,
		layers:  network.Layers(),
		cost:    cost,
	}

	layersCount := len(trainer.layers)
	trainer.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	trainer.potentialsPerLayer = make([][]float64, layersCount, layersCount)
	trainer.sp = make([][]float64, layersCount, layersCount)
	trainer.backward = make([][]float64, layersCount, layersCount)

	for l, layer := range trainer.layers {
		_, weightsCol, biasesCol := layer.Shapes()
		if l == 0 {
			trainer.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}
		if l == len(trainer.layers)-1 {
			trainer.outError = make([]float64, biasesCol, biasesCol)
		}

		trainer.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		trainer.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
		trainer.sp[l] = make([]float64, biasesCol, biasesCol)
		trainer.backward[l] = make([]float64, weightsCol, weightsCol)
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

	b.cost.CostDerivative(b.outError, b.acticationPerLayer[len(b.acticationPerLayer)-1], sample.Output)

	// Propagate output error to weights of output layer
	b.layers[layersCount-1].Activator().Derivative(b.sp[layersCount-1], b.potentialsPerLayer[len(b.potentialsPerLayer)-1])
	delta := mat.MulVectorElementWise(weightUpdates.Biases[layersCount-1], b.sp[layersCount-1], b.outError)

	mat.MulTransposeVector(weightUpdates.Weights[layersCount-1], delta, b.acticationPerLayer[len(b.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		lNo := layersCount - l
		potentials := b.potentialsPerLayer[len(b.potentialsPerLayer)-l]
		b.layers[lNo].Activator().Derivative(b.sp[lNo], potentials)

		b.layers[lNo+1].Backward(b.backward[lNo], delta)

		delta = mat.MulVectorElementWise(weightUpdates.Biases[lNo], b.backward[lNo], b.sp[lNo])
		mat.MulTransposeVector(weightUpdates.Weights[lNo], delta, b.acticationPerLayer[len(b.acticationPerLayer)-l-1])
	}
}

// CalculateCorrectness evaluates neural network across test samples to give averate cost and error rate
func CalculateCorrectness(nn Evaluator, cost Cost, samples []TrainExample) (avgCost float64, errors float64) {
	var sum float64
	var different float64

	for _, sample := range samples {
		output := nn.Evaluate(sample.Input)
		sum += cost.Cost(output, sample.Output)

		if mat.ArgMax(output) != mat.ArgMax(sample.Output) {
			different++
		}
	}

	avgCost = sum / float64(len(samples))
	errors = different / float64(len(samples))
	return
}
