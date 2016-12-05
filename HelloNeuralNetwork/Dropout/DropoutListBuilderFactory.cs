using System;

namespace HelloNeuralNetwork.Dropout
{
    public class DropoutListBuilderFactory
    {
        public IDropoutListBuilder Create(bool dropoutOn, Random random)
        {
            if (dropoutOn)
            {
                return new DropoutListBuilder(random);
            }
            
            return new FakeDropoutListBuilder();
        }
    }
}
