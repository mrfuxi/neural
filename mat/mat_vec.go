package mat

import (
	"math"
	"math/rand"
)

func CopyOfMatrix(src [][]float64) (dst [][]float64) {
	dst = make([][]float64, len(src))
	for i, row := range src {
		dst[i] = make([]float64, len(row))
		copy(dst[i], row)
	}
	return
}

func CopyOfVector(src []float64) (dst []float64) {
	dst = make([]float64, len(src))
	copy(dst, src)
	return
}

func SumVector(dst, src []float64) {
	for i, val := range src {
		dst[i] += val
	}
}

func SumMatrix(dst, src [][]float64) {
	for i, row := range src {
		SumVector(dst[i], row)
	}
}

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

func SubVector(dst, src []float64) {
	for i, val := range src {
		dst[i] -= val
	}
}

func SubMatrix(dst, src [][]float64) {
	for i, row := range src {
		SubVector(dst[i], row)
	}
}

func MulVectorByScalar(dst []float64, scalar float64) {
	for i, val := range dst {
		dst[i] = val * scalar
	}
}

func MulMatrixByScalar(dst [][]float64, scalar float64) {
	for _, row := range dst {
		MulVectorByScalar(row, scalar)
	}
}

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

func RandomVector(size int) []float64 {
	vector := make([]float64, size, size)
	for col := range vector {
		vector[col] = rand.NormFloat64()
	}

	return vector
}

func RandomMatrix(rows, cols int) [][]float64 {
	data := make([][]float64, rows)
	for row := range data {
		data[row] = RandomVector(cols)
	}
	return data
}

func MulVectorElementWise(dst, a, b []float64) []float64 {
	if dst == nil {
		dst = make([]float64, len(a), len(a))
	}
	for i := range dst {
		dst[i] = a[i] * b[i]
	}
	return dst
}

func SubVectorElementWise(a, b []float64) (diff []float64) {
	diff = make([]float64, len(a), len(a))
	for i := range diff {
		diff[i] = a[i] - b[i]
	}
	return
}

func VectorLen(a []float64) (vLen float64) {
	for _, val := range a {
		vLen += val * val
	}
	vLen = math.Sqrt(vLen)
	return
}

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

func ZeroVector(a []float64) {
	for i := range a {
		a[i] = 0
	}
}

func ZeroMatrix(a [][]float64) {
	for _, vec := range a {
		ZeroVector(vec)
	}
}

func ZeroVectorOfMatrixes(a [][][]float64) {
	for _, m := range a {
		ZeroMatrix(m)
	}
}
