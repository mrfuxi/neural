package neural_test

import (
	"testing"

	"github.com/mrfuxi/neural"
	"github.com/stretchr/testify/assert"
)

func TestFeedForward(t *testing.T) {
	input := []float64{1, 1}
	// expectedOutput := []float64{1, 1, 1}

	layer := neural.NewSimpleLayer(2, 3)
	nn := neural.NewNeuralNetwork(layer)
	output := nn.Evaluate(input)

	assert.Len(t, output, 3)
	// assert.Equal(t, expectedOutput, output)
}
