namespace HelloNeuralNetwork
{
    public class Parametrs
    {
        public int Epochs { get; set; }

        public int MiniBatchSize { get; set; }

        public double Eta { get; set; }

        public int NumberOfNeuronsInHiddenLayer { get; set; }

        public double Beta { get; set; }

        public double Lambda { get; set; }

        public int DataSize { get; set; }

        public bool DropoutOn {get; set; }

        public Parametrs(int epochs,
            int miniBatchSize, 
            double eta, 
            int numberOfNeuronsInHiddenLayer,
            double beta,
            double lambda,
            int dataSize,
            bool dropoutOn)
        {
            Epochs = epochs;
            MiniBatchSize = miniBatchSize;
            Eta = eta;
            NumberOfNeuronsInHiddenLayer = numberOfNeuronsInHiddenLayer;
            Beta = beta;
            Lambda = lambda;
            DataSize = dataSize;
            DropoutOn = dropoutOn;
        }
    }
}
