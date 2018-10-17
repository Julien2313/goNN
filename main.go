package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/goNN/models"
)

func main() {

	nbrInput := 1
	nbrHiddenLayer := 0
	nbrNeuronPerHiddenLayer := 0
	nbrOutput := 1

	rand.Seed(time.Now().UTC().UnixNano())
	var trainSet [][][]float64
	trainSet = make([][][]float64, 1000)

	for train := 0; train < len(trainSet); train++ {
		/*
			trainSet[train] = make([][]float64, 2)
			trainSet[train][0] = make([]float64, 3)
			trainSet[train][1] = make([]float64, 1)
			x := rand.Float64()*40.0 - 20.0
			y := rand.Float64()*40.0 - 20.0
			z := rand.Float64()*40.0 - 20.0
			a := x + y + z
			trainSet[train][0][0] = x
			trainSet[train][0][1] = y
			trainSet[train][0][2] = z
			trainSet[train][1][0] = a
		*/
		trainSet[train] = make([][]float64, 2)
		trainSet[train][0] = make([]float64, nbrInput)
		trainSet[train][1] = make([]float64, nbrOutput)
		x := rand.Float64()*40.0 - 20.0
		var a float64
		if x > 0 {
			a = 1.0
		} else {
			a = 0.0
		}
		trainSet[train][0][0] = x
		trainSet[train][1][0] = a

	}

	var nn models.NeuralNetwork

	//nn.Init(3, 1, 1, 32)
	nn.Init(nbrInput, nbrOutput, nbrHiddenLayer, nbrNeuronPerHiddenLayer)
	nn.SetInput(trainSet[999][0])
	nn.Propagate()
	fmt.Println(nn.Neurons[0][0].Value, nn.Neurons[1][0].Value)
	for x := 0; x < 1000; x++ {
		nn.Train(trainSet)
		if x == 0 {
			nn.Draw(x)
		}
	}
	fmt.Println(nn.Neurons[0][0].Value, nn.Neurons[1][0].Value)
	//nn.SetInput([]float64{5})
	//nn.Propagate()
	//fmt.Println(nn.Neurons[1][0].Value)

}
