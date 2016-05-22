package neural

import "github.com/mrfuxi/neural/mat"

// Trainer implements calculations of weights adjustments (WeightUpdates) in the network
// It operates on a single training example to prepare fractional result
type Trainer interface {
	Process(sample TrainExample, weightUpdates *WeightUpdates)
}

// TrainerFactory build Trainers. Multiple trainers will be created at the beginning of the training.
type TrainerFactory func(network Evaluator, cost CostDerivative) Trainer

type trainer struct {
	network Evaluator
	layers  []Layer
	cost    CostDerivative

	acticationPerLayer [][]float64
	potentialsPerLayer [][]float64
	outError           []float64
	sp                 [][]float64
	backward           [][]float64
}

// NewBackpropagationTrainer builds new trainer that uses backward propagation algorithm
func NewBackpropagationTrainer(network Evaluator, cost CostDerivative) Trainer {
	t := trainer{
		network: network,
		layers:  network.Layers(),
		cost:    cost,
	}

	layersCount := len(t.layers)
	t.acticationPerLayer = make([][]float64, layersCount+1, layersCount+1)
	t.potentialsPerLayer = make([][]float64, layersCount, layersCount)
	t.sp = make([][]float64, layersCount, layersCount)
	t.backward = make([][]float64, layersCount, layersCount)

	for l, layer := range t.layers {
		_, weightsCol, biasesCol := layer.Shapes()
		if l == 0 {
			t.acticationPerLayer[0] = make([]float64, weightsCol, weightsCol)
		}
		if l == len(t.layers)-1 {
			t.outError = make([]float64, biasesCol, biasesCol)
		}

		t.acticationPerLayer[l+1] = make([]float64, biasesCol, biasesCol)
		t.potentialsPerLayer[l] = make([]float64, biasesCol, biasesCol)
		t.sp[l] = make([]float64, biasesCol, biasesCol)
		if l > 0 {
			t.backward[l-1] = make([]float64, weightsCol, weightsCol)
		}
	}

	return &t
}

// Process executes backward propagation algorithm to get weight updates
func (t *trainer) Process(sample TrainExample, weightUpdates *WeightUpdates) {
	layersCount := len(t.layers)
	lNo := layersCount - 1

	copy(t.acticationPerLayer[0], sample.Input)
	for l, layer := range t.layers {
		layer.Forward(t.potentialsPerLayer[l], t.acticationPerLayer[l])
		layer.Activator().Activation(t.acticationPerLayer[l+1], t.potentialsPerLayer[l])
	}

	t.cost.CostDerivative(
		weightUpdates.Biases[lNo],
		t.acticationPerLayer[len(t.acticationPerLayer)-1],
		sample.Output,
		t.potentialsPerLayer[len(t.potentialsPerLayer)-1],
		t.layers[lNo].Activator(),
	)

	// Propagate output error to weights of output layer
	delta := weightUpdates.Biases[lNo]
	mat.MulTransposeVector(weightUpdates.Weights[lNo], delta, t.acticationPerLayer[len(t.acticationPerLayer)-2])

	for l := 2; l <= layersCount; l++ {
		lNo = layersCount - l
		potentials := t.potentialsPerLayer[len(t.potentialsPerLayer)-l]
		t.layers[lNo].Activator().Derivative(t.sp[lNo], potentials)

		t.layers[lNo+1].Backward(t.backward[lNo], delta)

		delta = mat.MulVectorElementWise(weightUpdates.Biases[lNo], t.backward[lNo], t.sp[lNo])
		mat.MulTransposeVector(weightUpdates.Weights[lNo], delta, t.acticationPerLayer[len(t.acticationPerLayer)-l-1])
	}
}
