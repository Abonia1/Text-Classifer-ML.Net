using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Transforms;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Runtime.Api;

namespace Model
{
    public class Program
    {

        

        static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.tsv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.tsv");
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");

        //task used with async to save the model in .zip before the task get finish
        static async Task Main(string[] args)
        {
           
           var model = await Train();
            //await Evaluate(Train.model);
            Evaluate(model);
           Predict(model);



            //return Task<TResult>;
        }

        //train the model
        public static async Task<PredictionModel<SentimentData, SentimentPrediction>> Train()
        {
            //Instance used to load,process,featurize the data
            var pipeline = new LearningPipeline();

            //to load train data
            pipeline.Add(new TextLoader(_dataPath).CreateFrom<SentimentData>(useHeader: true));

            pipeline.Add(new Dictionarizer("Label"));
            // TextFeaturizer to convert the SentimentText column into a numeric vector called Features used by the ML algorithm
            pipeline.Add(new TextFeaturizer("Features", "SentimentText"));

            //choose learning algorithm
            pipeline.Add(new StochasticDualCoordinateAscentClassifier());
            //pipeline.Add(new LogisticRegressionClassifier());
            //pipeline.Add(new NaiveBayesClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            
            //pipeline.Add(new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs =2});

            //train the model
            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            //save model
            await model.WriteAsync(_modelpath);
            return model;

        }

        //test the model
        public static void Evaluate(PredictionModel<SentimentData, SentimentPrediction> model)
        {
           

            //load the test data 
            var testData = new TextLoader(_testDataPath).CreateFrom<SentimentData>(useHeader: true);

            //computes the quality metrics for the PredictionModel
            var evaluator = new ClassificationEvaluator();

            //to get metrices computed by binary classification evaluator
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");
            Console.WriteLine($"Accuracy: {metrics.AccuracyMacro:P2}");
            //Console.WriteLine($"Auc: {metrics.Auc:P2}");
           // Console.WriteLine($"F1Score: {metrics.F1Score:P2}");


        }


        //predicting the model
        public static void Predict(PredictionModel<SentimentData, SentimentPrediction> model)
        {
         
            //prediction based on comments
            IEnumerable<SentimentData> sentiments = new[]

 {
                new SentimentData
    {
        SentimentText = "Saisie d'une absence non-rémunérée sur 2 heures "

    },
     new SentimentData
    {
        SentimentText = "Est ce qu une absence pour accident du travail a un impact sur l attribution des RTTCONGES ANNUELS / RTT / COMPTE EPARGNE TEMPSb"

    },
     new SentimentData
    {
        SentimentText = "New hr is on board"

    },
      new SentimentData
    {
        SentimentText = "Peut on saisir une demi journée de grève ?"

    }
    /*new SentimentData
    {
        SentimentText = "HR Systems"

    },
     new SentimentData
    {
        SentimentText = "Personnal Data"

    },
      new SentimentData
    {
        SentimentText = "Travel & Expenses"

    },
       new SentimentData
    {
        SentimentText = "Benefits & Pay"

    },
    new SentimentData
    {
        SentimentText = "Housing and loan"

    },
     new SentimentData
    {
        SentimentText = "Career & Talent"

    },
      new SentimentData
    {
        SentimentText = "Onboarding"

    },
       new SentimentData
    {
        SentimentText = "Career & Talent"

    },
        new SentimentData
    {
        SentimentText = "Leave of Absence"

    },
         new SentimentData
    {
        SentimentText = "Retirement & Savings"

    },
          new SentimentData
    {
        SentimentText = "Career & Talent"

    },
           new SentimentData
    {
        SentimentText = "Onboarding"

    },
            new SentimentData
    {
        SentimentText = "Career & Talent"

    }*/
};
            //sentiment prediction
            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);
            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
            var sentimentsAndPredictions = sentiments.Zip(predictions, (sentiment, prediction) => (sentiment, prediction));
            foreach (var item in sentimentsAndPredictions)
            {
                Console.WriteLine($"SentimentText: {item.sentiment.SentimentText} | Category: {(item.prediction.Category )}");
            }
            Console.WriteLine();
            

        }
    }

}




