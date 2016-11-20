using MathNet.Numerics.LinearAlgebra;

namespace HelloNeuralNetwork.Models
{
    public class DigitImage
    {
        public Matrix<double> Pixels { get; }
        

        public DigitImage(double[,] pixels)
        {
            Pixels = Matrix<double>.Build.DenseOfArray(pixels);
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < 28; ++i)
            {
                for (int j = 0; j < 28; ++j)
                {
                    if (Pixels.At(i,j) == 0)
                        s += " "; // white
                    else if (Pixels.At(i,j) == 255)
                        s += "O"; // black
                    else
                        s += "."; // gray
                }
                s += "\n";
            }
            return s;
        } // ToString

    }
}
