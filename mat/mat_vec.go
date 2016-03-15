package mat

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

func MulTransposeVector(a, b []float64) (dst [][]float64) {
	dst = make([][]float64, len(a))
	for i, valA := range a {
		dst[i] = make([]float64, len(b))
		for j, valB := range b {
			dst[i][j] = valA * valB
		}
	}
	return
}
