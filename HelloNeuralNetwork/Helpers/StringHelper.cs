namespace HelloNeuralNetwork.Helpers
{
    public class StringHelper
    {
        public string CleanFileNameFromExtension(string fileName)
        {
            var dotIndex = fileName.IndexOf('.');

            return fileName.Substring(0, dotIndex);
        }

    }
}
