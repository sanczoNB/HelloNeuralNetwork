using HelloNeuralNetwork.Helpers;
using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Models
{
    public class DigitImageForStuding : DigitImage
    {

        private readonly MatrixHelpers _helpers = new MatrixHelpers();

        public Matrix<double> Label { get; }

        public DigitImageForStuding(double[,] pixels, byte label) : base(pixels)
        {
            Label = _helpers.VectorizedResult(label);
        }

        public override string ToString()
        {
            var s = base.ToString();

            s += Label.ToString();

            return s;
        }
    }
}
