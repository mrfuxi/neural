package neural

import "math"

// Activator calculates neuron activation and it's derivative from given potential
type Activator interface {
	Activation(float64) float64
	Derivative(float64) float64
}

// NewLinearActivator creates Activator that applies linear function to given potential
// Actication: a*potential + 0
// Derivative: a
func NewLinearActivator(a float64) Activator {
	return &linearActication{a}
}

type linearActication struct {
	a float64
}

func (l *linearActication) Activation(potential float64) float64 {
	return l.a * potential
}

func (l *linearActication) Derivative(potential float64) float64 {
	return l.a
}

// NewSigmoidActivator creates Activator that applies linear function to given potential
// Actication: a*potential + 0
// Derivative: a
func NewSigmoidActivator() Activator {
	return &sigmoidActivator{}
}

type sigmoidActivator struct{}

func (s *sigmoidActivator) Activation(potential float64) float64 {
	return 1.0 / (1.0 + math.Exp(-potential))
}

func (s *sigmoidActivator) Derivative(potential float64) float64 {
	activation := s.Activation(potential)
	return activation * (1 - activation)
}

// NewStepFunction creates Activator that returns 0 or 1 only
// Actication: 1 if potential >= 0 else 0
// Derivative: 0 (is that correct?)
func NewStepFunction() Activator {
	return &stepActicator{}
}

type stepActicator struct{}

func (s *stepActicator) Activation(potential float64) float64 {
	if potential >= 0 {
		return 1
	}
	return 0
}

func (s *stepActicator) Derivative(potential float64) float64 {
	return 0 // is that correct?
}
