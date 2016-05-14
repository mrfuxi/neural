package neural

import "math"

// Activator calculates neuron activation and it's derivative from given potential
type Activator interface {
	Activation(dst, potentials []float64)
	Derivative(dst, potentials []float64)
}

// NewLinearActivator creates Activator that applies linear function to given potential
//
// Actication: a*potential + 0
//
// Derivative: a
func NewLinearActivator(a float64) Activator {
	return &linearActication{a}
}

type linearActication struct {
	a float64
}

func (l *linearActication) Activation(dst, potentials []float64) {
	for i, potential := range potentials {
		dst[i] = l.a * potential
	}
}

func (l *linearActication) Derivative(dst, potentials []float64) {
	for i := range potentials {
		dst[i] = l.a
	}
}

// NewSigmoidActivator creates Activator that applies linear function to given potential
//
// Actication: 1/(1+exp(-potential))
//
// Derivative: f(potential) * (1- f(potential))
func NewSigmoidActivator() Activator {
	return &sigmoidActivator{}
}

type sigmoidActivator struct{}

func (s *sigmoidActivator) Activation(dst, potentials []float64) {
	for i, potential := range potentials {
		dst[i] = 1.0 / (1.0 + math.Exp(-potential))
	}
}

func (s *sigmoidActivator) Derivative(dst, potentials []float64) {
	s.Activation(dst, potentials)
	for i, activation := range dst {
		dst[i] = activation * (1 - activation)
	}
}

// NewStepActivator creates Activator that returns 0 or 1 only.
//
// Actication: 1 if potential >= 0 else 0
//
// Derivative: 1 (is that correct?)
func NewStepActivator() Activator {
	return &stepActicator{}
}

type stepActicator struct{}

func (s *stepActicator) Activation(dst, potentials []float64) {
	for i, potential := range potentials {
		if potential >= 0 {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

func (s *stepActicator) Derivative(dst, potentials []float64) {
	for i := range potentials {
		dst[i] = 1
	}
}

// NewSoftmaxActivator creates Activator that scales responses in layer from 0 to 1.
//
// Sum of responses in layer are equal 1, so it can be interpret as probability.
// This activator should be used in last layer with Log Likelihood.
// Derivative is not implemented as it should not be needed. If used it will panic.
func NewSoftmaxActivator() Activator {
	return &softmaxActicator{}
}

type softmaxActicator struct{}

func (s *softmaxActicator) Activation(dst, potentials []float64) {
	var sum float64

	for i, potential := range potentials {
		dst[i] = math.Exp(potential)
		sum += dst[i]
	}

	for i, dstVal := range dst {
		dst[i] = dstVal / sum
	}
}

func (s *softmaxActicator) Derivative(dst, potentials []float64) {
	panic("Derivative of Softmax should not be used in ANN")
}

// NewTanhActivator creates Activator that returns values between -1 and 1.
// Very similar to sigmoid function in nature.
//
// Actication: tanh(potential)
//
// Derivative: 1/f(potential/2)/2
func NewTanhActivator() Activator {
	return &tanhActicator{}
}

type tanhActicator struct{}

func (s *tanhActicator) Activation(dst, potentials []float64) {
	for i, potential := range potentials {
		dst[i] = math.Tanh(potential)
	}
}

func (s *tanhActicator) Derivative(dst, potentials []float64) {
	for i, potential := range potentials {
		dst[i] = (1 + math.Tanh(potential/2)) / 2
	}
}
