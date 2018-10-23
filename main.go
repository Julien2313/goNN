package main

import (
	"fmt"
	"math/rand"

	"github.com/goNN/models"
)

func main() {

	nbrInput := 1
	nbrHiddenLayer := 1
	nbrNeuronPerHiddenLayer := 1
	nbrOutput := 1
	seed := int64(4) //time.Now().UTC().UnixNano()
	fmt.Println(seed)
	rand.Seed(seed)
	var trainSet [][][]float64
	var sizeTrainData int = 10000
	trainSet = make([][][]float64, sizeTrainData)

	for train := 0; train < len(trainSet); train++ {
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

	nn.Init(nbrInput, nbrOutput, nbrHiddenLayer, nbrNeuronPerHiddenLayer)
	nn.SetInput(trainSet[sizeTrainData-1][0])
	nn.Propagate()
	nn.Draw(0)
	fmt.Println(nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)
	fmt.Println("Error : = ", nn.CheckTraining(trainSet))
	var nbrMaxEpoch int = 10000
	for epoch := 0; epoch < nbrMaxEpoch; epoch++ {
		nn.Train(trainSet)
		// if epoch%100 == 0 {
		// 	fmt.Println("Error : = ", nn.CheckTraining(trainSet))
		// }
	}

	nn.Draw(nbrMaxEpoch - 1)
	fmt.Println("Error : = ", nn.CheckTraining(trainSet))
	fmt.Println(nn.Neurons[0][0].Value, nn.Neurons[1][0].Value, nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)
	fmt.Println(nn.Neurons[1][0].Weights[0], nn.Neurons[1][0].Biais)

}
