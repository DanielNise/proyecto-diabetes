document.getElementById("prediction-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    // Obtener los datos del formulario
    const data = {
        Pregnancies: parseFloat(document.getElementById("Pregnancies").value),
        Glucose: parseFloat(document.getElementById("Glucose").value),
        BloodPressure: parseFloat(document.getElementById("BloodPressure").value),
        SkinThickness: parseFloat(document.getElementById("SkinThickness").value),
        Insulin: parseFloat(document.getElementById("Insulin").value),
        BMI: parseFloat(document.getElementById("BMI").value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById("DiabetesPedigreeFunction").value),
        Age: parseFloat(document.getElementById("Age").value),
    };

    // Cargar el modelo de TensorFlow.js
    const model = await tf.loadLayersModel('modelo_tensorflowjs/model.json');

    // Convertir los datos en un tensor
    const input = tf.tensor2d([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]]);

    // Realizar la predicción
    const prediction = model.predict(input);

    // Obtener el valor de la predicción (0 o 1)
    const predictionValue = prediction.dataSync()[0];

    // Mostrar el resultado
    document.getElementById("prediction").textContent = `Predicción: ${predictionValue === 1 ? 'Diabetes' : 'Sano'}`;

    // Mostrar la probabilidad de cada clase
    const probability = model.predict(input).dataSync();
    document.getElementById("probability").textContent = `Probabilidades: [No Diabetes: ${(1 - probability[0]).toFixed(2)}, Diabetes: ${probability[0].toFixed(2)}]`;
});
