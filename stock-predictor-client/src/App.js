import React, { useEffect, useState } from "react";

function App() {
  const [prediction, setPrediction] = useState(null); // state variable for prediction
  const [loading, setLoading] = useState(true); // state variable for loading

  useEffect(() => {
    fetch("http://127.0.0.1:5000/predict")  //fetch prediction from flask backend
      .then((res) => res.json())  //parse response as json
      .then((data) => {
        setPrediction(data);  //set prediction state variable
        setLoading(false);  //set loading state variable to false
      })
      .catch((err) => {//if error catching prediction
        console.error("Error fetching prediction:", err);
        setLoading(false);
      });
  }, []);

  return (  //return jsx, user interface we see on the screen
    <div style={{ padding: "2rem", fontFamily: "Arial" }}>
      <h1>📈 S&P 500 Market Prediction</h1>
      {loading ? (  //if loading is true, show loading message
        <p>Loading prediction...</p>
      ) : prediction ? (  //if prediction is true (loading has finished), show prediction message
        <>
          <p><strong>Prediction:</strong> {prediction.prediction}</p>
          <p><strong>Confidence:</strong> {(prediction.probability * 100).toFixed(2)}%</p>
        </>
      ) : (  //if prediction is false, show failed to load prediction message
        <p>Failed to load prediction.</p>
      )}
    </div>
  );
}

export default App;