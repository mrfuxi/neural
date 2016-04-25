package neural

// Cost interface represents way of calculating neural network cost
// Cost method should calculate cost of single example.
// It does not account for normalization. Normalization factor of 1/n should be applied further on
type Cost interface {
	Cost(output, desired []float64) float64
}

//CostDerivative method should calculate derivative of cost of single example
type CostDerivative interface {
	CostDerivative(dst, output, desired, potentials []float64, activator Activator)
}

type QuadraticCost struct{}

func (q *QuadraticCost) Cost(output, desired []float64) float64 {
	sum := 0.0
	for i, out := range output {
		diff := desired[i] - out
		sum += diff * diff
	}
	return 0.5 * sum
}

func (q *QuadraticCost) CostDerivative(dst, output, desired, potentials []float64, activator Activator) {
	activator.Derivative(dst, potentials)

	for i, out := range output {
		dst[i] = (out - desired[i]) * dst[i]
	}
}

func NewQuadraticCost() *QuadraticCost {
	return &QuadraticCost{}
}
