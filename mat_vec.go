package neural

func copyOfMatrix(src [][]float64) (dst [][]float64) {
	dst = make([][]float64, len(src))
	for i, row := range src {
		dst[i] = make([]float64, len(row))
		copy(dst[i], row)
	}
	return
}

func copyOfVector(src []float64) (dst []float64) {
	dst = make([]float64, len(src))
	copy(dst, src)
	return
}

func sumVector(dst, src []float64) {
	for i, val := range src {
		dst[i] += val
	}
}

func sumMatrix(dst, src [][]float64) {
	for i, row := range src {
		sumVector(dst[i], row)
	}
}

func subVector(dst, src []float64) {
	for i, val := range src {
		dst[i] -= val
	}
}

func subMatrix(dst, src [][]float64) {
	for i, row := range src {
		subVector(dst[i], row)
	}
}

func mulVectorByScalar(dst []float64, scalar float64) {
	for i, val := range dst {
		dst[i] = val * scalar
	}
}

func mulMatrixByScalar(dst [][]float64, scalar float64) {
	for _, row := range dst {
		mulVectorByScalar(row, scalar)
	}
}

func mulTransposeVector(a, b []float64) (dst [][]float64) {
	dst = make([][]float64, len(a), len(a))
	for i, valA := range a {
		dst[i] = make([]float64, len(b), len(b))
		for j, valB := range b {
			dst[i][j] = valA * valB
		}
	}
	return
}
