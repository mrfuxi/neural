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
	nnSaveFile = flag.String("save-file", "", "Save neural network to file")
	nnLoadFile = flag.String("load-file", "", "Load neural network to file")
	validate   = flag.Bool("validate", false, "validate against test data")
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

func loadTestData() ([]neural.TrainExample, []neural.TrainExample) {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	trainData := prepareMnistData(train)
	testData := prepareMnistData(test)
	return trainData, testData
}

func main() {
	trainData, testData := loadTestData()

	activator := neural.NewSigmoidActivator()
	nn := neural.NewNeuralNetwork(
		activator,
		[]int{inputSize, 100, 10},
		neural.NewFullyConnectedLayer,
		neural.NewFullyConnectedLayer,
	)

	flag.Parse()

	if *nnLoadFile != "" {
		fn, err := os.Open(*nnLoadFile)
		if err != nil {
			log.Fatalln(err)
		}
		if err := neural.Load(nn, fn); err != nil {
			log.Fatalln(err)
		}
	}

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	t0 := time.Now()
	neural.Train(nn, trainData, 1, 10, 4, neural.NewBackwardPropagationTrainer)
	dt := time.Since(t0)

	fmt.Println("Training complete in", dt)
	if *validate {
		different := 0
		for _, sample := range testData {
			output := nn.Evaluate(sample.Input)
			if mat.ArgMax(output) != mat.ArgMax(sample.Output) {
				different++
			}
		}
		success := 100 * float64(different) / float64(len(testData))
		fmt.Printf("Error: %.2f%% \n", success)
	}

	if *nnSaveFile != "" {
		fn, err := os.OpenFile(*nnSaveFile, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			log.Fatalln(err)
		}
		if err := neural.Save(nn, fn); err != nil {
			log.Fatalln(err)
		}
	}
}
