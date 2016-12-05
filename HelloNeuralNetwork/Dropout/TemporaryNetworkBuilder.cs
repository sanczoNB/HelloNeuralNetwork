using System;
using HelloNeuralNetwork.Dropout;
using HelloNeuralNetwork.Models;

namespace HelloNeuralNetwork
{
    public class TemporaryNetworkBuilder
    {
        private readonly DropoutAdapter _adapter;

        private readonly IDropoutListBuilder _builder;

        public TemporaryNetworkBuilder(DropoutAdapter adapter, Random random, IDropoutListBuilder dropoutListBuilder)
        {
            _adapter = adapter;
            _builder = dropoutListBuilder;
        }

        public TemporaryNetwork Build(Network mainNetwork)
        {
            var activeNeurons = _builder.BuildList(mainNetwork.HiddenLayerSize());

            var tempNet = new TemporaryNetwork(activeNeurons);

            var weightForHiddenLayer =
                _adapter.CreateNewWeightsMatrixBeetwenInputAndHiddenLayers(
                    mainNetwork.GetWeightsBeetwenInputAndHiddenLayer(), activeNeurons);
            var biasOnHiddenLayer =
                _adapter.CreateNewBiasOnHiddenLayer(mainNetwork.GetBiasOnHiddenLayer(), activeNeurons);
            var weightForOutputLayer =
                _adapter.CreateNewWeightsMatrixBeetwenHiddenAndOutputLayer(
                    mainNetwork.GetWeightsBeetwenHiddenAndOutputLayer(), activeNeurons);
            var biasOnOutputLayer = mainNetwork.GetBiasOnOutputLayer();

            tempNet.SetWeightsBeetwenInputAndHiddenLayer(weightForHiddenLayer);
            tempNet.SetBiasOnHiddenLayer(biasOnHiddenLayer);
            tempNet.SetWeightsBeetwenHiddenAndOutputLayer(weightForOutputLayer);
            tempNet.SetBiasOnOutputLayer(biasOnOutputLayer);

            return tempNet;
        }
    }
}
