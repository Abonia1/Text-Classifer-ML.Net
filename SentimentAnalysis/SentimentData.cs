using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace Model
{



    public class SentimentData
    {
        [Column(ordinal: "0")]
        public string SentimentText;
        [Column(ordinal: "1", name: "Label")]
        public string Category;
    }

    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Category;
    }
}
