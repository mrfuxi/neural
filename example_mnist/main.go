package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"github.com/mrfuxi/neural"
	"github.com/mrfuxi/neural/mat"
	"github.com/petar/GoMNIST"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
	inputSize  = GoMNIST.Width * GoMNIST.Height
)

func prepareMnistData(rawData *GoMNIST.Set) []neural.TrainExample {
	trainData := make([]neural.TrainExample, rawData.Count())
	for i := range trainData {
		image, label := rawData.Get(i)
		trainData[i].Input = make([]float64, inputSize, inputSize)
		trainData[i].Output = make([]float64, 10, 10)
		for j, pix := range image {
			trainData[i].Input[j] = (float64(pix)/255)*0.9 + 0.1
		}

		for j := range trainData[i].Output {
			trainData[i].Output[j] = 0.1
		}
		trainData[i].Output[label] = 0.9
	}
	return trainData
}

func main() {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	trainData := prepareMnistData(train)
	testData := prepareMnistData(test)

	hiddenLayer1 := neural.NewSimpleLayer(inputSize, 100)
	outLayer := neural.NewSimpleLayer(100, 10)

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(activator, hiddenLayer1, outLayer)

	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	t0 := time.Now()

	nn.Train(trainData, 1, 10, 4)
	dt := time.Since(t0)

	different := 0
	for _, sample := range testData {
		output := nn.Evaluate(sample.Input)
		if mat.ArgMax(output) != mat.ArgMax(sample.Output) {
			different++
		}
	}
	success := 100 * float64(different) / float64(len(testData))
	fmt.Printf("\nTraining complete in %v: %.2f%% \n", dt, success)
}
