using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace AutoMLMultiClassBug
{
    class Program
    {
        class TaxiTrip
        {
            [LoadColumn(0)]
            public string VendorId;

            [LoadColumn(1)]
            public float RateCode;

            [LoadColumn(2)]
            public float PassengerCount;

            [LoadColumn(3)]
            public float TripTimeInSeconds;

            [LoadColumn(4)]
            public float TripDistance;

            [LoadColumn(5)]
            public string PaymentType;

            [LoadColumn(6)]
            public float FareAmount;
        }

        class TaxiTripFarePrediction
        {
            [ColumnName("Score")]
            public float[] Payment;
        }

        static void Main(string[] args)
        {
            MulticlassClassificationExperiment.Run();
        }

        public static class MulticlassClassificationExperiment
        {
            private static string TrainDataPath = "taxi-fare-train.csv";
            private static string TestDataPath = "taxi-fare-test.csv";
            private static string ModelPath = @"OptDigitsModel.zip";
            private static string LabelColumnName = "PaymentType";
            private static uint ExperimentTime = 5;

            public static void Run()
            {
                MLContext mlContext = new MLContext();

                // STEP 1: Load data
                IDataView trainDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, separatorChar: ',', hasHeader: true);
                IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, separatorChar: ',', hasHeader:true);

                // STEP 2: Run AutoML experiment
                Console.WriteLine($"Running AutoML multiclass classification experiment for {ExperimentTime} seconds...");
                ExperimentResult<MulticlassClassificationMetrics> experimentResult = mlContext.Auto()
                    .CreateMulticlassClassificationExperiment(ExperimentTime)
                    .Execute(trainDataView, LabelColumnName);

                // STEP 3: Print metric from the best model
                RunDetail<MulticlassClassificationMetrics> bestRun = experimentResult.BestRun;
                Console.WriteLine($"Total models produced: {experimentResult.RunDetails.Count()}");
                Console.WriteLine($"Best model's trainer: {bestRun.TrainerName}");
                Console.WriteLine($"Metrics of best model from validation data --");
                PrintMetrics(bestRun.ValidationMetrics);

                // STEP 6: Create prediction engine from the best trained model
                var predictionEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(bestRun.Model);

                // STEP 7: Initialize new pixel data, and get the predicted number
                var testTaxiTripData = new TaxiTrip
                {
                    VendorId = "CMT",
                    RateCode = 1,
                    PassengerCount = 1,
                    TripTimeInSeconds = 1271,
                    TripDistance = 3.8F,
                    PaymentType = "CRD",
                    FareAmount = 17.5F,
                };
                var prediction = predictionEngine.Predict(testTaxiTripData);
                Console.WriteLine($"Predicted number for test data:");

                foreach ( var x in prediction.Payment)
                {
                    Console.WriteLine(x.ToString());
                }

                Console.WriteLine("Press any key to continue...");
                Console.ReadKey();
            }

            private static void PrintMetrics(MulticlassClassificationMetrics metrics)
            {
                Console.WriteLine($"LogLoss: {metrics.LogLoss}");
                Console.WriteLine($"LogLossReduction: {metrics.LogLossReduction}");
                Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
                Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
            }
        }
    }
}
