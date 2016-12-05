using System.Collections.Generic;

namespace HelloNeuralNetwork.Dropout
{
    public interface IDropoutListBuilder
    {
        List<int> BuildList(int count);
    }
}
