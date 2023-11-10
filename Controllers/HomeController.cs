using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using Microsoft.ML;
using System.Diagnostics;

using practica_n4.Models;

namespace practica_n4.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly PredictionEngine<SentimentModel.ModelInput, SentimentModel.ModelOutput> _predictionEngine;

        public HomeController(ILogger<HomeController> logger)
        {
            _logger = logger;

            // Configurar el motor de predicción con el modelo entrenado
            var mlContext = new MLContext();
            var mlModel = mlContext.Model.Load(SentimentModel.MLNetModelPath, out var _);
            _predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentModel.ModelInput, SentimentModel.ModelOutput>(mlModel);
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
public IActionResult AnalyzeSentiment(string text)
{
    // Realizar la predicción de sentimiento
    var input = new SentimentModel.ModelInput { Col0 = text };
    var prediction = SentimentModel.Predict(input);

    // Almacenar el resultado en ViewBag para mostrar en la vista
    ViewBag.Sentiment = prediction.PredictedLabel;

    // Volver a la vista Index para mostrar el resultado
    return View("Index");
}

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}


