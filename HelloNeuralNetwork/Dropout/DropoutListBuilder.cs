using System;
using System.Collections.Generic;
using System.Linq;

namespace HelloNeuralNetwork.Dropout
{
    public class DropoutListBuilder : IDropoutListBuilder
    {
        private readonly Random _random;

        public DropoutListBuilder(Random random)
        {
            this._random = random;
        }

        public List<int> BuildList(int count)
        {


            var dropOutCandidate = Enumerable.Range(0, count).ToList();

            var winnerCount = count/2;

            var dropWinnerList = new List<int>(winnerCount);

            for (var i = 0; i <winnerCount; i++)
            {
                var winnerIndex = (int)Math.Floor(_random.NextDouble()*dropOutCandidate.Count);

                var winner = dropOutCandidate[winnerIndex];
                dropOutCandidate.RemoveAt(winnerIndex);
                dropWinnerList.Add(winner);
            }

            dropWinnerList.Sort();

            return dropWinnerList;
         }

    }
}
