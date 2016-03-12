package main

import (
	"flag"
	"log"
	"os"
	"runtime/pprof"

	"github.com/mrfuxi/neural"
	"github.com/petar/GoMNIST"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func main() {
	train, _, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	inputSize := GoMNIST.Width * GoMNIST.Height

	trainData := make([]neural.TrainExample, train.Count())
	for i := range trainData {
		image, label := train.Get(i)
		trainData[i].Input = make([]float64, inputSize, inputSize)
		trainData[i].Output = make([]float64, 10, 10)
		for j, pix := range image {
			trainData[i].Input[j] = float64(pix) / 255
		}
		trainData[i].Output[label] = 1.0
	}

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

	nn.Train(trainData[:1000], 5, 100, 3)
}
