using System;
using System.Linq;

namespace HelloNeuralNetwork
{
    class Simulator
    {
        public int NumOfNeuronsInHidenLayer = 30;

        public IActivationFunction ActivationFunction = new SigmoidFunction();

        public int Epochs = 30;

        public int MiniBatchSize = 5;

        public double Eta = 3;

        public void SimpleLearnig()
        {
            var sizes = new int[] {784, NumOfNeuronsInHidenLayer, 10};

            var network = new Network(sizes, ActivationFunction);

            var dataLoader = new DataLoader();

            var traingData = dataLoader.LoadForStuding();

            var evaluatingData = dataLoader.LoadForEvaluating();

/*            for (var i = 0; i < 784; i++)
            {
                Console.WriteLine(evaluatingData.First().Pixels.At(i,0));
            }*/

            network.Sgd(traingData, Epochs, MiniBatchSize, Eta, evaluatingData);

        }

    }
}
