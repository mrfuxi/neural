package mat

import (
	"math"
	"math/rand"
)

// CopyOfMatrix makes a deep copy of matrix (slice of slices of float64)
func CopyOfMatrix(src [][]float64) (dst [][]float64) {
	dst = make([][]float64, len(src))
	for i, row := range src {
		dst[i] = make([]float64, len(row))
		copy(dst[i], row)
	}
	return
}

// CopyOfVector makes a deep copy of vector (slice of float64)
func CopyOfVector(src []float64) (dst []float64) {
	dst = make([]float64, len(src))
	copy(dst, src)
	return
}

// SumVector adds values from src slice to dst slice
func SumVector(dst, src []float64) {
	for i, val := range src {
		dst[i] += val
	}
}

// SumMatrix adds values from src matrix to dst matrix
func SumMatrix(dst, src [][]float64) {
	for i, row := range src {
		SumVector(dst[i], row)
	}
}

// SumVectors set dst values as sum of src slices
func SumVectors(dst []float64, srcs ...[]float64) {
	cols := len(dst)
	sum := 0.0
	for i := 0; i < cols; i++ {
		sum = 0
		for _, src := range srcs {
			sum += src[i]
		}
		dst[i] = sum
	}
}

// SumMatrixes set dst values as sum of src matrixes
func SumMatrixes(dst [][]float64, srcs ...[][]float64) {
	rowsCount := len(dst)
	rows := make([][]float64, len(srcs), len(srcs))
	for r := 0; r < rowsCount; r++ {
		for i, src := range srcs {
			rows[i] = src[r]
		}
		SumVectors(dst[r], rows...)
	}
}

// SubVector subtracts src value from dst
func SubVector(dst, src []float64) {
	for i, val := range src {
		dst[i] -= val
	}
}

// SubMatrix subtracts src value from dst
func SubMatrix(dst, src [][]float64) {
	for i, row := range src {
		SubVector(dst[i], row)
	}
}

// MulVectorByScalar multiplies every values in dst by a constant factor
func MulVectorByScalar(dst []float64, scalar float64) {
	for i, val := range dst {
		dst[i] = val * scalar
	}
}

// MulMatrixByScalar multiplies every values in dst by a constant factor
func MulMatrixByScalar(dst [][]float64, scalar float64) {
	for _, row := range dst {
		MulVectorByScalar(row, scalar)
	}
}

// MulTransposeVector multiplies two matrices a' and b and places them to dst
func MulTransposeVector(dst [][]float64, a, b []float64) [][]float64 {
	if dst == nil {
		dst = make([][]float64, len(a))
		for i := range a {
			dst[i] = make([]float64, len(b))
		}
	}

	for i, valA := range a {
		row := dst[i]
		for j, valB := range b {
			row[j] = valA * valB
		}
	}
	return dst
}

// RandomVector creates vector of given size.
// Values are distributes using normal distribution
func RandomVector(size int) []float64 {
	vector := make([]float64, size, size)
	for col := range vector {
		vector[col] = rand.NormFloat64()
	}

	return vector
}

// RandomMatrix creates matrix of given size.
// Values are distributes using normal distribution
func RandomMatrix(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for row := range data {
		data[row] = RandomVector(cols)
	}
	return data
}

// MulVectorElementWise multiplies a by b value by value.
// Result is set to dst
func MulVectorElementWise(dst, a, b []float64) []float64 {
	if dst == nil {
		dst = make([]float64, len(a), len(a))
	}
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
	return dst
}

// SubVectorElementWise subtracts b from a (a-b).
// Result is retuned as dst
func SubVectorElementWise(dst, a, b []float64) {
	for i := range dst {
		dst[i] = a[i] - b[i]
	}
	return
}

// VectorLen calculates euclidean length of the vector
func VectorLen(a []float64) (vLen float64) {
	for _, val := range a {
		vLen += val * val
	}
	vLen = math.Sqrt(vLen)
	return
}

// ArgMax calculates argmax(a)
func ArgMax(a []float64) int {
	maxVal := math.SmallestNonzeroFloat64
	maxArg := -1

	for i, val := range a {
		if val > maxVal {
			maxVal = val
			maxArg = i
		}
	}
	return maxArg
}

// ZeroVector sets all values to 0
func ZeroVector(a []float64) {
	for i := range a {
		a[i] = 0
	}
}

// ZeroMatrix sets all values to 0
func ZeroMatrix(a [][]float64) {
	for _, vec := range a {
		ZeroVector(vec)
	}
}

// ZeroVectorOfMatrixes sets all values to 0
func ZeroVectorOfMatrixes(a [][][]float64) {
	for _, m := range a {
		ZeroMatrix(m)
	}
}
