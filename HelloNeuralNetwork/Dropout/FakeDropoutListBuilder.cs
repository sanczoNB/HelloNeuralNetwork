using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HelloNeuralNetwork.Dropout
{
    class FakeDropoutListBuilder : IDropoutListBuilder
    {
        public List<int> BuildList(int count)
        {
            return Enumerable.Range(0, count).ToList();
        }
    }
}
