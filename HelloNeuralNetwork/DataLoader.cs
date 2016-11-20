using System;
using System.Collections.Generic;
using System.IO;
using HelloNeuralNetwork.Models;

namespace HelloNeuralNetwork
{
    public class DataLoader
    {
        public string TestLabesAdress = @"C:\data\t10k-labels.idx1-ubyte";

        public string TestDataAddress = @"C:\data\t10k-images.idx3-ubyte";

        public string TraningLabesAdress = @"C:\data\train-labels.idx1-ubyte";

        public string TraningDataAddress = @"C:\data\train-images.idx3-ubyte";

        public int SizeOfTraingData = 50000;

        public int SizeOfEvaluatingData = 10000;

        public List<DigitImageForStuding> LoadForStuding()
        { 
            var results = new List<DigitImageForStuding>();
            try { 
            FileStream ifsLabels =
             new FileStream(TraningLabesAdress,
             FileMode.Open); // test labels
            FileStream ifsImages =
             new FileStream(TraningDataAddress,
             FileMode.Open); // test images

            BinaryReader brLabels =
             new BinaryReader(ifsLabels);
            BinaryReader brImages =
             new BinaryReader(ifsImages);

            int magic1 = brImages.ReadInt32(); // discard
            int numImages = brImages.ReadInt32();
            int numRows = brImages.ReadInt32();
            int numCols = brImages.ReadInt32();

            int magic2 = brLabels.ReadInt32();
            int numLabels = brLabels.ReadInt32();

            double[,] pixels = new double[784, 1];

            // each test image
            for (int di = 0; di < SizeOfTraingData; ++di)
            {
                for (int i = 0; i < 784; ++i)
                {
                        byte b = brImages.ReadByte();

                        pixels[i, 0] = b / 255.0;
                    }

                byte lbl = brLabels.ReadByte();

                results.Add(new DigitImageForStuding(pixels, lbl));
                  
            } // each image

            ifsImages.Close();
            brImages.Close();
            ifsLabels.Close();
            brLabels.Close();

            
        }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }

            return results;
        }

        public List<DigitImageForEvaluating> LoadForEvaluating()
        {
            var results = new List<DigitImageForEvaluating>();
            try
            {
                FileStream ifsLabels =
                 new FileStream(TestLabesAdress,
                 FileMode.Open); // test labels
                FileStream ifsImages =
                 new FileStream(TestDataAddress,
                 FileMode.Open); // test images

                BinaryReader brLabels =
                 new BinaryReader(ifsLabels);
                BinaryReader brImages =
                 new BinaryReader(ifsImages);

                int magic1 = brImages.ReadInt32(); // discard
                int numImages = brImages.ReadInt32();
                int numRows = brImages.ReadInt32();
                int numCols = brImages.ReadInt32();

                int magic2 = brLabels.ReadInt32();
                int numLabels = brLabels.ReadInt32();

                double[,] pixels = new double[784, 1];

                // each test image
                for (int di = 0; di < SizeOfEvaluatingData; ++di)
                {
                    for (int i = 0; i < 784; ++i)
                    {   
                            byte b = brImages.ReadByte();
                            pixels[i, 0] = b / 255.0;
                    }

                    byte lbl = brLabels.ReadByte();

                    results.Add(new DigitImageForEvaluating(pixels, lbl));

                } // each image

                ifsImages.Close();
                brImages.Close();
                ifsLabels.Close();
                brLabels.Close();


            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }

            return results;
        }
    }
}
