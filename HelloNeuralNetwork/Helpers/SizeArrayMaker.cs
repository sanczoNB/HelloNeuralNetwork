using System.Security.Cryptography.X509Certificates;

namespace HelloNeuralNetwork.Helpers
{
    public class SizeArrayMaker
    {

        public int FirstLayerCount => 784;

        public virtual int LastLayerCount { get;}

        public int[] Make(int hiddenLayerCount)
        {
            return new int[] { FirstLayerCount, hiddenLayerCount, LastLayerCount };
        }
    }
}
