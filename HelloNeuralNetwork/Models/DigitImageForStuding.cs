using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Models
{
    public class DigitImageForStuding : DigitImage
    {

        public Matrix<double> Label { get; }

        public DigitImageForStuding(double[,] pixels, byte label) : base(pixels)
        {
            Label = VectorizedResult(label);
        }

        private Matrix<double> VectorizedResult(byte label)
        {
            var vectorized = Matrix<double>.Build.Dense(10, 1);
            vectorized[label, 0] = 1;

            return vectorized;
        }

        public override string ToString()
        {
            var s = base.ToString();

            s += Label.ToString();

            return s;
        }
    }
}
