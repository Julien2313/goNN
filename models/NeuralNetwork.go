package models

import (
	"math/rand"
	"errors"
	"fmt"
	"math"
)

type NeuralNetwork struct {
	NbrInput, NbrOutput, NbrHiddenLayers, NbrNeuronsPerLayer int
	LearningRate float64
	Neurons [][]Neuron
	OutputWaited []float64
}

func (nn *NeuralNetwork) Init(nbrInput, nbrOutput, nbrHiddenLayers, nbrNeuronsPerLayer int) {
	nn.NbrInput = nbrInput
	nn.NbrOutput = nbrOutput
	nn.NbrHiddenLayers = nbrHiddenLayers
	nn.NbrNeuronsPerLayer = nbrNeuronsPerLayer

	nn.LearningRate = 0.1

	//Init Neurons
	nn.Neurons = make([][]Neuron, 2 + nbrHiddenLayers)
	nn.Neurons[0] = make([]Neuron, nbrInput)

	for numNeuron := 0; numNeuron < nbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Biais = rand.Float64() * 40.0 - 20.0
	}

	for numLayer := 1; numLayer < nbrHiddenLayers+1; numLayer++ {
		nn.Neurons[numLayer] = make([]Neuron, nbrNeuronsPerLayer)

		for numNeuron := 0; numNeuron < nbrNeuronsPerLayer; numNeuron++ {
			nn.Neurons[numLayer][numNeuron].Biais = rand.Float64() * 40.0 - 20.0

			if numLayer == 1 {
				nn.Neurons[numLayer][numNeuron].Weights = make([]float64, nbrInput)
				for numWeightNeuron := 0; numWeightNeuron < nbrInput; numWeightNeuron++ {
					nn.Neurons[numLayer][numNeuron].Weights[numWeightNeuron] = rand.Float64()*40.0 - 20.0
				}
			} else {
				nn.Neurons[numLayer][numNeuron].Weights = make([]float64, nbrNeuronsPerLayer)
				for numWeightNeuron := 0; numWeightNeuron < nbrNeuronsPerLayer; numWeightNeuron++ {
					nn.Neurons[numLayer][numNeuron].Weights[numWeightNeuron] = rand.Float64()*40.0 - 20.0
				}
			}
		}
	}

	nn.Neurons[nbrHiddenLayers+1] = make([]Neuron, nbrOutput)

	for numNeuron := 0; numNeuron < nbrOutput; numNeuron++ {
		nn.Neurons[nbrHiddenLayers+1][numNeuron].Biais = rand.Float64() * 40.0 - 20.0

		nn.Neurons[nbrHiddenLayers+1][numNeuron].Weights = make([]float64, nbrNeuronsPerLayer)
		for numWeightNeuron := 0; numWeightNeuron < nbrNeuronsPerLayer; numWeightNeuron++ {
			nn.Neurons[nbrHiddenLayers+1][numNeuron].Weights[numWeightNeuron] = rand.Float64() * 40.0 - 20.0
		}

	}
}

func (nn *NeuralNetwork) Print () {
	for _, layer := range nn.Neurons {
		for _, neuron := range layer {
			fmt.Print(neuron.Value, neuron.Weights, ", ")
		}
		fmt.Println()
		fmt.Println("----------------")
	}
}

func (nn *NeuralNetwork) SetInput(inputs []float64) error{
	if len(inputs) != nn.NbrInput {
		return errors.New("Number of inputs doesn't match the model")
	}

	for numNeuron := 0; numNeuron < nn.NbrInput; numNeuron++ {
		nn.Neurons[0][numNeuron].Value = inputs[numNeuron]
	}

	return nil
}

func (nn *NeuralNetwork) Propagate() {
	for numLayer := 1; numLayer < len(nn.Neurons); numLayer++ {
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			sum := nn.Neurons[numLayer][numNeuron].Biais
			for numWeightNeuron := 0; numWeightNeuron < len(nn.Neurons[numLayer - 1]); numWeightNeuron++ {
				sum += nn.Neurons[numLayer][numNeuron].Weights[numWeightNeuron] * nn.Neurons[numLayer-1][numWeightNeuron].Value
			}
			nn.Neurons[numLayer][numNeuron].Sigmoid(sum)
		}
	}
}

func (nn *NeuralNetwork) ComputeError(output []float64) {
	for numNeuronOutput := 0; numNeuronOutput < nn.NbrOutput; numNeuronOutput++ {
		nn.Neurons[nn.NbrHiddenLayers + 1][numNeuronOutput].Error = math.Pow(nn.Neurons[nn.NbrHiddenLayers + 1][numNeuronOutput].Value - output[numNeuronOutput], 2 )
	}
	/*
	for numLayer := nn.NbrHiddenLayers; numLayer >= 0; numLayer++ {
		for numNeuron := 0; numNeuron < len(nn.Neurons[numLayer]); numNeuron++ {
			nn.Neurons[numLayer][numNeuron].Error = math.Pow(nn.Neurons[numLayer][numNeuron].Value - output[numNeuron], 2 )
		}

	}*/
}


func (nn *NeuralNetwork) Learn() {
	for numNeuronOutput := 0; numNeuronOutput < nn.NbrOutput; numNeuronOutput++ {
	}
}


func (nn *NeuralNetwork) Train(dataSet [][][]float64) {
	for _, data := range dataSet {
		nn.SetInput(data[0])
		nn.Propagate()
		nn.ComputeError(data[1])
		nn.Learn()
	}
}
