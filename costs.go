package neural

import (
	"math"

	"github.com/mrfuxi/neural/mat"
)

// Cost interface represents way of calculating neural network cost
// Cost method should calculate cost of single example.
// It does not account for normalization. Normalization factor of 1/n should be applied further on
type Cost interface {
	Cost(output, desired []float64) float64
}

// CostDerivative method should calculate derivative of cost of single example
type CostDerivative interface {
	CostDerivative(dst, output, desired, potentials []float64, activator Activator)
}

// CostCostDerrivative represents both way of calculating neural network cost
// as well as it's derivative (delta)
type CostCostDerrivative interface {
	Cost
	CostDerivative
}

type quadraticCost struct{}

func (q *quadraticCost) Cost(output, desired []float64) float64 {
	sum := 0.0
	for i, out := range output {
		diff := desired[i] - out
		sum += diff * diff
	}
	return 0.5 * sum
}

func (q *quadraticCost) CostDerivative(dst, output, desired, potentials []float64, activator Activator) {
	activator.Derivative(dst, potentials)

	for i, out := range output {
		dst[i] = (out - desired[i]) * dst[i]
	}
}

// NewQuadraticCost creates quadratic cost function also known as mean squared error or just MSE
func NewQuadraticCost() CostCostDerrivative {
	return &quadraticCost{}
}

type corssEntropyCost struct{}

func (c *corssEntropyCost) Cost(output, desired []float64) float64 {
	sum := 0.0
	for i, out := range output {
		y := desired[i]
		sum -= y*math.Log(out) + (1-y)*math.Log(1-out)
	}
	return 0.5 * sum
}

func (c *corssEntropyCost) CostDerivative(dst, output, desired, potentials []float64, activator Activator) {
	for i, out := range output {
		dst[i] = out - desired[i]
	}
}

// NewCrossEntropyCost creates cross entropy cost function.
// Comparing with quadratic cost it's derivative is not affected by activation function derivative.
// That means learning process is faster and avoids saturation of sigmoid function.
// It should be used together with sigmoid activation function in the last layer.
// In case of using it with different activator CostDerivative is no longer correct.
func NewCrossEntropyCost() CostCostDerrivative {
	return &corssEntropyCost{}
}

type logLikelihoodCost struct{}

func (c *logLikelihoodCost) Cost(output, desired []float64) float64 {
	arg := mat.ArgMax(desired)
	return -math.Log(output[arg])
}

func (c *logLikelihoodCost) CostDerivative(dst, output, desired, potentials []float64, activator Activator) {
	for i, out := range output {
		dst[i] = out - desired[i]
	}
}

// NewLogLikelihoodCost creates cross entropy cost function.
// Similar to cross entropy function it's faster than quadratic cost function,
// however should be used with Softmax activator in last layer for math to be correct.
func NewLogLikelihoodCost() CostCostDerrivative {
	return &logLikelihoodCost{}
}
