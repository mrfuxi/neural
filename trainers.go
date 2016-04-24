package neural

import "github.com/mrfuxi/neural/mat"

// Trainer implements calculations of weights adjustments (WeightUpdates) in the network
// It operates on a single training example to prepare fractional result
type Trainer interface {
	Process(sample TrainExample, weightUpdates *WeightUpdates)
}

// TrainerFactory build Trainers. Multiple trainers will be created at the beginning of the training.
type TrainerFactory func(network Evaluator) Trainer

type backwardPropagationTrainerBase struct {
	network Evaluator
	layers  []Layer

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
	outError           []float64
	sp                 [][]float64
	backward           [][]float64
}

func (b *backwardPropagationTrainerBase) PrepareBuffors(network Evaluator) {
	b.network = network
	b.layers = network.Layers()

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

type quadraticCostTrainer struct {
	backwardPropagationTrainerBase
	QuadraticCost
}

// NewQuadraticCostTrainer builds new trainer that uses backward propagation algorithm
func NewQuadraticCostTrainer(network Evaluator) Trainer {
	trainer := quadraticCostTrainer{}
	trainer.PrepareBuffors(network)
	return &trainer
}

// Process executes backward propagation algorithm to get weight updates
func (q *quadraticCostTrainer) Process(sample TrainExample, weightUpdates *WeightUpdates) {
	layersCount := len(q.layers)

	copy(q.acticationPerLayer[0], sample.Input)
	for l, layer := range q.layers {
		layer.Forward(q.potentialsPerLayer[l], q.acticationPerLayer[l])
		layer.Activator().Activation(q.acticationPerLayer[l+1], q.potentialsPerLayer[l])
	}

	q.CostDerivative(q.outError, q.acticationPerLayer[len(q.acticationPerLayer)-1], sample.Output)

	// Propagate output error to weights of output layer
	q.layers[layersCount-1].Activator().Derivative(q.sp[layersCount-1], q.potentialsPerLayer[len(q.potentialsPerLayer)-1])
	delta := mat.MulVectorElementWise(weightUpdates.Biases[layersCount-1], q.sp[layersCount-1], q.outError)

	mat.MulTransposeVector(weightUpdates.Weights[layersCount-1], delta, q.acticationPerLayer[len(q.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		lNo := layersCount - l
		potentials := q.potentialsPerLayer[len(q.potentialsPerLayer)-l]
		q.layers[lNo].Activator().Derivative(q.sp[lNo], potentials)

		q.layers[lNo+1].Backward(q.backward[lNo], delta)

		delta = mat.MulVectorElementWise(weightUpdates.Biases[lNo], q.backward[lNo], q.sp[lNo])
		mat.MulTransposeVector(weightUpdates.Weights[lNo], delta, q.acticationPerLayer[len(q.acticationPerLayer)-l-1])
	}
}
