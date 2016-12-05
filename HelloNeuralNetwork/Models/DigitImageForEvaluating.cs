namespace HelloNeuralNetwork.Models
{
    public class DigitImageForEvaluating : DigitImage
    {

        public byte Label { get; }

        public DigitImageForEvaluating(double[,] pixels, byte label) : base(pixels)
        {
            Label = label;
        }

        public override string ToString()
        {
            var s = base.ToString();
            s += Label.ToString();
            return s;
        }
    }
}
