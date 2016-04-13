package neural_test

import (
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestLinearActivator(t *testing.T) {
	testMatrix := []struct {
		a                          float64
		in, activation, derivative []float64
	}{
		{a: 2, in: []float64{-1}, activation: []float64{-2}, derivative: []float64{2}},
		{a: 2, in: []float64{0}, activation: []float64{0}, derivative: []float64{2}},
		{a: 2, in: []float64{1}, activation: []float64{2}, derivative: []float64{2}},

		{a: -2, in: []float64{-1}, activation: []float64{2}, derivative: []float64{-2}},
		{a: -2, in: []float64{0}, activation: []float64{0}, derivative: []float64{-2}},
		{a: -2, in: []float64{1}, activation: []float64{-2}, derivative: []float64{-2}},

		{a: 0, in: []float64{-1}, activation: []float64{0}, derivative: []float64{0}},
		{a: 0, in: []float64{0}, activation: []float64{0}, derivative: []float64{0}},
		{a: 0, in: []float64{1}, activation: []float64{0}, derivative: []float64{0}},
	}

	for _, example := range testMatrix {
		activator := neural.NewLinearActivator(example.a)
		activation := make([]float64, len(example.in), len(example.in))
		derivative := make([]float64, len(example.in), len(example.in))
		activator.Activation(activation, example.in)
		activator.Derivative(derivative, example.in)
		assert.Equal(t, example.activation, activation)
		assert.Equal(t, example.derivative, derivative)
	}
}

func TestSigmoidActivator(t *testing.T) {
	testMatrix := []struct {
		in, activation, derivative []float64
	}{
		{[]float64{-2}, []float64{0.11920}, []float64{0.104994}},
		{[]float64{0}, []float64{0.5}, []float64{0.25}},
		{[]float64{2}, []float64{0.88079}, []float64{0.104994}},
	}

	activator := neural.NewSigmoidActivator()
	for _, example := range testMatrix {
		activation := make([]float64, len(example.in), len(example.in))
		derivative := make([]float64, len(example.in), len(example.in))
		activator.Activation(activation, example.in)
		activator.Derivative(derivative, example.in)
		assert.InDeltaSlice(t, example.activation, activation, 0.00001)
		assert.InDeltaSlice(t, example.derivative, derivative, 0.00001)
	}
}
