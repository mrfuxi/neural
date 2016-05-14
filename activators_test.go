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

func TestSoftmaxActivator(t *testing.T) {
	testMatrix := []struct {
		in, activation []float64
	}{
		{[]float64{1, 1, 1}, []float64{0.3333, 0.3333, 0.3333}},
		{[]float64{0, 1, 0}, []float64{0.2119, 0.5761, 0.2119}},
		{[]float64{1, 2, 3}, []float64{0.0900, 0.2447, 0.6652}},
		{[]float64{0, 0, 0}, []float64{0.3333, 0.3333, 0.3333}},
		{[]float64{-1, 0, 1}, []float64{0.0900, 0.2447, 0.6652}},
		{[]float64{1, 1, 10}, []float64{0.0001, 0.0001, 0.9998}},
	}

	activator := neural.NewSoftmaxActivator()
	for _, example := range testMatrix {
		activation := make([]float64, len(example.in), len(example.in))
		activator.Activation(activation, example.in)
		assert.InDeltaSlice(t, example.activation, activation, 0.0001)

		// Sum up to 1
		sum := 0.0
		for _, val := range activation {
			sum += val
		}
		assert.InDelta(t, 1.0, sum, 0.000001)

		// Derivative not panics
		assert.Panics(t, func() {
			activator.Derivative(activation, example.in)
		})
	}
}

func TestTanhActivator(t *testing.T) {
	testMatrix := []struct {
		in, activation, derivative []float64
	}{
		{[]float64{-2}, []float64{-0.96402758}, []float64{0.119202922}},
		{[]float64{0}, []float64{0}, []float64{0.5}},
		{[]float64{2}, []float64{0.96402758}, []float64{0.880797078}},
		{[]float64{-2, 0, 2}, []float64{-0.96402758, 0, 0.96402758}, []float64{0.119202922, 0.5, 0.880797078}},
	}

	activator := neural.NewTanhActivator()
	for _, example := range testMatrix {
		activation := make([]float64, len(example.in), len(example.in))
		derivative := make([]float64, len(example.in), len(example.in))
		activator.Activation(activation, example.in)
		activator.Derivative(derivative, example.in)
		assert.InDeltaSlice(t, example.activation, activation, 0.00001)
		assert.InDeltaSlice(t, example.derivative, derivative, 0.00001)
	}
}
