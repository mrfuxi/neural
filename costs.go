package neural

// Cost represents way of calculating neural network cost
type Cost interface {
	// Cost(dst, output, desired []float64)
	CostDerivative(dst, output, desired []float64)
}

type quadraticCost struct{}

func (q *quadraticCost) Cost(dst, output, desired []float64) {
	// norm := 0.5 / float64(len(output))
	// sum := 0.0
	// for i, out := range output {
	// 	diff := desired[i] - out
	// 	sum += diff * diff
	// }
	// return norm * sum
}

func (q *quadraticCost) CostDerivative(dst, output, desired []float64) {
	for i, out := range output {
		dst[i] = out - desired[i]
	}
}

func NewQuadraticCost() Cost {
	return &quadraticCost{}
}
