using HelloNeuralNetwork.Models;

namespace HelloNeuralNetwork.Dropout
{
    public class NetworkUpdater
    {
        public void Update(Network network, TemporaryNetwork tempNet)
        {
            UpdateWeightsForHiddenLayer(network, tempNet);
            UpdateBiasForHiddenLayer(network, tempNet);
            UpdateWeightsForOutputLayer(network, tempNet);
            UpdateBiasForOutputLayer(network, tempNet);
            
        }

        private void UpdateWeightsForHiddenLayer(Network network, TemporaryNetwork tempNet)
        {
            for (var i = 0; i < tempNet.ActiveNeuronInHiddenLayerCount(); i++)
            {
                var neuronLabel = tempNet.GetActiveNeuronLabel(i);
                var weight = tempNet.GetWeightsForNeuronFromHiddenLayer(i);

                network.SetWeightsForNeuronInHiddenLayer(neuronLabel, weight);
            }
        }

        private void UpdateBiasForHiddenLayer(Network network, TemporaryNetwork tempNet)
        {
            for (var i = 0; i < tempNet.ActiveNeuronInHiddenLayerCount(); i++)
            {
                var neuronLabel = tempNet.GetActiveNeuronLabel(i);
                var bias = tempNet.GetBiasForNeuronFromHiddenLayer(i);

                network.SetBiasOnHiddenLayer(neuronLabel, bias);
            }
        }

        private void UpdateWeightsForOutputLayer(Network network, TemporaryNetwork tempNet)
        {
            for (var i = 0; i < tempNet.ActiveNeuronInHiddenLayerCount(); i++)
            {
                var neuronLabel = tempNet.GetActiveNeuronLabel(i);
                var weights = tempNet.GetWeightsForInputFromHiddenLayer(i);

                network.SetWeightsForInputsFromHiddenLayer(neuronLabel, weights);
            }
        }

        private void UpdateBiasForOutputLayer(Network network, TemporaryNetwork tempNet)
        {
            for (var i = 0; i < tempNet.ActiveNeuronInHiddenLayerCount(); i++)
            {
                var biases = tempNet.GetBiasOnOutputLayer();

                network.SetBiasesOnOutputLayer(biases);
            }
        }
    }
}
