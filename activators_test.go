package neural_test

import (
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestLinearActivator(t *testing.T) {
	testMatrix := []struct {
		a                          float64
		in, activation, derivative float64
	}{
		{2, -1, -2, 2},
		{2, 0, 0, 2},
		{2, 1, 2, 2},

		{-2, -1, 2, -2},
		{-2, 0, 0, -2},
		{-2, 1, -2, -2},

		{0, -1, 0, 0},
		{0, 0, 0, 0},
		{0, 1, 0, 0},
	}

	for _, example := range testMatrix {
		activator := neural.NewLinearActivator(example.a)
		activation := activator.Activation(example.in)
		derivative := activator.Derivative(example.in)
		assert.Equal(t, example.activation, activation)
		assert.Equal(t, example.derivative, derivative)
	}
}

func TestSigmoidActivator(t *testing.T) {
	testMatrix := []struct {
		in, activation, derivative float64
	}{
		{-2, 0.11920, 0.104994},
		{0, 0.5, 0.25},
		{2, 0.88079, 0.104994},
	}

	activator := neural.NewSigmoidActivator()
	for _, example := range testMatrix {
		activation := activator.Activation(example.in)
		derivative := activator.Derivative(example.in)
		assert.InDelta(t, example.activation, activation, 0.00001)
		assert.InDelta(t, example.derivative, derivative, 0.00001)
	}
}
