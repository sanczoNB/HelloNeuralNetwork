using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace HelloNeuralNetwork.IO
{
    public class FileSaver
    {
        public void SaveImages(List<Bitmap> images, string folderPath )
        {
            for (var i = 0; i < images.Count; i++)
            {
                var image = images[i];
                image.Save(folderPath+$"neuron {i+1}.jpg");
                image.Dispose();
            }
        }

        public Bitmap  Merge(List<Color[,]> images)
        {
            var imageCount = images.Count;

            var imageResolutionX = images.First().GetLength(0);
            var imageResolutionY = images.First().GetLength(1);

            var xGab = 5;
            var yGab = 5;

            var rowAndColumnCount = (int)Math.Ceiling(Math.Sqrt(imageCount));
            var xSize = rowAndColumnCount*imageResolutionX + (rowAndColumnCount - 1)*xGab;
            var ysize = rowAndColumnCount*imageResolutionY + (rowAndColumnCount - 1)*yGab;

            var bigPicture= new Bitmap(xSize, ysize);
            

            //Koloruje wszystko na jeden kolor
            for (var row = 0; row < ysize; row++)
            {
                for (var column = 0; column < xSize; column++)
                {
                    bigPicture.SetPixel(column, row, Color.White);
                }
            }

            for (var pictureNumber = 0; pictureNumber < images.Count; pictureNumber++)
            {
                var rowNumber = pictureNumber/rowAndColumnCount;
                var columnNumber = pictureNumber%rowAndColumnCount;

                var x = columnNumber*imageResolutionX + columnNumber*xGab;
                var y = rowNumber*imageResolutionY + rowNumber*yGab;
                PutPicture(images[pictureNumber],bigPicture, x, y, imageResolutionX, imageResolutionY);
            }

            return bigPicture;
        }

        private void PutPicture(Color[,] smallPicture, Bitmap bigPicture, int x, int y, int imageResolutionX, int imageResolutionY)
        {
            for (var c = 0; c <imageResolutionX; c++)
            {
                for (var r = 0; r < imageResolutionY; r++)
                {
                    var bx = c + x;
                    var by = r + y;
                    bigPicture.SetPixel(bx, by, smallPicture[c,r]);
                }
            }
        }
    }
}
