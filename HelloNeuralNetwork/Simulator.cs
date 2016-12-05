using System;
using System.Linq;
using HelloNeuralNetwork.Helpers;

namespace HelloNeuralNetwork
{
    class Simulator
    {
        public int NumOfNeuronsInHidenLayer = 30;

        public IActivationFunction ActivationFunction = new SigmoidFunction();

        public int Epochs = 1;

        public int MiniBatchSize = 1;

        public double Eta = 3;

        private readonly ParamReader _paramReader = new ParamReader();

        private readonly DataLoader _dataLoader = new DataLoader();

        private int _firstLayerCount = 784;

        private int _lastLayerCount = 10;

        private FileLoader _loader = new FileLoader();

        public void LearningForDiffrentSetsOfParam()
        {
            var @params = _paramReader.Read();

            var traingData = _dataLoader.LoadForStuding();

            var evaluatingData = _dataLoader.LoadForEvaluating();

            int counter = 0;

            var sizeArrayMaker = new ClassicSizeArrayMaker();

            foreach (var param in @params)
            {
                counter++;
                Console.WriteLine("Wykonuje dla {0}-etgo zestawu danych", counter);

                var sizes = sizeArrayMaker.Make(param.NumberOfNeuronsInHiddenLayer);

                var network = new Network(sizes, ActivationFunction, param.DropoutOn);

                network.Sgd(traingData.Take(param.DataSize).ToList(), param, evaluatingData);
            }
        }

        public int Evaluate(string filePath, int hiddenLayer)
        {
           var network = new Network(MakeSizeTab(hiddenLayer), new SigmoidFunction(), false);

            var weightsAndBiases = _loader.LoadFromBinary(filePath);

            var weights = weightsAndBiases[0];
            var biases = weightsAndBiases[1];

            network.SetWeights(weights);
            network.SetBiases(biases);

            var evaluatingData = _dataLoader.LoadForEvaluating();

            return network.Evaluate(evaluatingData);
        }

    }
}
