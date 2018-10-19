package main

import (
	"fmt"
	"math/rand"

	"github.com/goNN/models"
)

func main() {

	nbrInput := 1
	nbrHiddenLayer := 1
	nbrNeuronPerHiddenLayer := 32
	nbrOutput := 1
	seed := int64(2) //time.Now().UTC().UnixNano()
	fmt.Println(seed)
	rand.Seed(seed)
	var trainSet [][][]float64
	trainSet = make([][][]float64, 10000)

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
		if x > -13 {
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
	nn.SetInput(trainSet[9999][0])
	nn.Propagate()
	fmt.Println(nn.Neurons[0][0].Value, nn.Neurons[1][0].Value, nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)
	fmt.Println("Error : = ", nn.CheckTraining(trainSet))
	for x := 0; x < 1000; x++ {
		nn.Train(trainSet)
		if x == 0 {
			nn.Draw(x)
		}
		if x == 999 {
			nn.Draw(x)
		}
		// if x%10 == 0 {
		// 	fmt.Println(nn.CheckTraining(trainSet))
		// }
	}

	fmt.Println("Error : = ", nn.CheckTraining(trainSet))
	fmt.Println(nn.Neurons[0][0].Value, nn.Neurons[1][0].Value, nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)
	fmt.Println(nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)

}
