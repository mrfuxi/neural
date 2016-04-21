package neural

import "github.com/mrfuxi/neural/mat"

// Trainer implements calculations of weights adjustments (WeightUpdates) in the network
// It operates on a single training example to prepare fractional result
type Trainer interface {
	Process(sample TrainExample, weightUpdates *WeightUpdates)
}

// TrainerFactory build Trainers. Multiple trainers will be created at the beginning of the training.
type TrainerFactory func(network Evaluator, cost Cost) Trainer

type backwardPropagationTrainerBase struct {
	network Evaluator
	layers  []Layer
	cost    Cost

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
	outError           []float64
	sp                 [][]float64
	backward           [][]float64
}

func (b *backwardPropagationTrainerBase) PrepareBuffors(network Evaluator, cost Cost) {
	b.network = network
	b.layers = network.Layers()
	b.cost = cost

	layersCount := len(b.layers)
	b.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	b.potentialsPerLayer = make([][]float64, layersCount, layersCount)
	b.sp = make([][]float64, layersCount, layersCount)
	b.backward = make([][]float64, layersCount, layersCount)

	for l, layer := range b.layers {
		_, weightsCol, biasesCol := layer.Shapes()
		if l == 0 {
			b.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}
		if l == len(b.layers)-1 {
			b.outError = make([]float64, biasesCol, biasesCol)
		}

		b.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		b.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
		b.sp[l] = make([]float64, biasesCol, biasesCol)
		b.backward[l] = make([]float64, weightsCol, weightsCol)
	}
}

type backwardPropagationTrainer struct {
	backwardPropagationTrainerBase
}

// NewBackwardPropagationTrainer builds new trainer that uses backward propagation algorithm
func NewBackwardPropagationTrainer(network Evaluator, cost Cost) Trainer {
	trainer := backwardPropagationTrainer{}
	trainer.PrepareBuffors(network, cost)
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
