package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"time"

	"github.com/mrfuxi/neural"
	"github.com/petar/GoMNIST"
)

var (
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
	nnSaveFile = flag.String("save-file", "", "Save neural network to file")
	nnLoadFile = flag.String("load-file", "", "Load neural network to file")
	inputSize  = GoMNIST.Width * GoMNIST.Height
)

func prepareMnistData(rawData *GoMNIST.Set) []neural.TrainExample {
	trainData := make([]neural.TrainExample, rawData.Count())
	for i := range trainData {
		image, label := rawData.Get(i)
		trainData[i].Input = make([]float64, inputSize, inputSize)
		trainData[i].Output = make([]float64, 10, 10)
		for j, pix := range image {
			trainData[i].Input[j] = (float64(pix) / 255)
		}

		for j := range trainData[i].Output {
			trainData[i].Output[j] = 0
		}
		trainData[i].Output[label] = 1
	}
	return trainData
}

func loadTestData() ([]neural.TrainExample, []neural.TrainExample, []neural.TrainExample) {
	train, test, err := GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}

	tmp := prepareMnistData(train)
	trainData := tmp[:40000]
	validationData := tmp[40000:]
	testData := prepareMnistData(test)
	return trainData, validationData, testData
}

func epocheCallback(nn neural.Evaluator, cost neural.Cost, validationData, testData []neural.TrainExample) neural.EpocheCallback {
	return func(epoche int, dt time.Duration) {
		_, validationErrors := neural.CalculateCorrectness(nn, cost, validationData)
		_, testErrors := neural.CalculateCorrectness(nn, cost, testData)
		if epoche == 1 {
			fmt.Println("epoche,validation error,test error")
		}
		fmt.Printf("%v,%v,%v\n", epoche, validationErrors, testErrors)
	}
}

func main() {
	trainData, validationData, testData := loadTestData()

	activator := neural.NewSigmoidActivator()
	outActivator := neural.NewSoftmaxActivator()
	nn := neural.NewNeuralNetwork(
		[]int{inputSize, 100, 10},
		neural.NewFullyConnectedLayer(activator),
		neural.NewFullyConnectedLayer(outActivator),
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

	cost := neural.NewLogLikelihoodCost()
	options := neural.TrainOptions{
		Epochs:         100,
		MiniBatchSize:  10,
		LearningRate:   0.4,
		Regularization: 5,
		TrainerFactory: neural.NewBackpropagationTrainer,
		EpocheCallback: epocheCallback(nn, cost, validationData, testData),
		Cost:           cost,
	}

	t0 := time.Now()
	neural.Train(nn, trainData, options)
	dt := time.Since(t0)

	fmt.Println("Training complete in", dt)

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
