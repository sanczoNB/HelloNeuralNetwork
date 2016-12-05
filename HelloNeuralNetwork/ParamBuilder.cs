using System.Collections.Generic;

namespace HelloNeuralNetwork
{
    public class ParamBuilder
    {
        private string _epochsKey = "epochs";

        private string _miniBatchKey = "miniBatch";

        private string _etaKey = "eta";

        private string _hiddenLayerKey = "hiddenLayer";

        private string _betaKey = "beta";

        private string _lambdaKey = "lambda";

        private string _trainingDataSizeKey = "n";

        private string _dropoutOnKey = "dropoutOn";

        private readonly int _defaultEpochs = -1;

        private readonly int _defaultMiniBatch = -1;

        private readonly double _defaultEta = 0.0;

        private readonly int _defaultHiddenLayer = -1;

        private readonly double _defaultBeta = 0.0;

        private readonly double _defaultLambda = 0.0;

        private readonly int _defaultDataSize = 50000;

        public Parametrs Build(Dictionary<string, string> dictionary)
        {
            string word;
           
            var epochs = dictionary.TryGetValue(_epochsKey, out word) ? int.Parse(word) : _defaultEpochs;

            var miniBatch = dictionary.TryGetValue(_miniBatchKey, out word) ? int.Parse(word) : _defaultMiniBatch;

            var eta = dictionary.TryGetValue(_etaKey, out word) ? double.Parse(word) : _defaultEta;

            var hiddenLayer = dictionary.TryGetValue(_hiddenLayerKey, out word) ? int.Parse(word) : _defaultHiddenLayer;

            var beta = dictionary.TryGetValue(_betaKey, out word) ? double.Parse(word) : _defaultBeta;

            var lambda = dictionary.TryGetValue(_lambdaKey, out word) ? double.Parse(word) : _defaultLambda;

            var dataSize = (dictionary.TryGetValue(_trainingDataSizeKey, out word) ? int.Parse(word) : _defaultDataSize);

            var dropoutOn = dictionary.TryGetValue(_dropoutOnKey, out word);

            return new Parametrs(epochs,miniBatch, eta, hiddenLayer, beta, lambda, dataSize, dropoutOn);

        }

    }
}
