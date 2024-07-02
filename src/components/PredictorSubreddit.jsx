import React, { useState } from 'react'
import axios from 'axios';

const PredictorSubreddit = (props) => {
  const [subreddit, setSubreddit] = useState("")
  const [orientation, setOrientation] = useState("")
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const handleSubmit = (e) => {
      e.preventDefault();
      setLoading(true);
      setError(false);
      var data = {
        'subreddit': subreddit
      }
  const backendUrl = process.env.REACT_APP_BACKEND_URL;

    axios.post(`${backendUrl}/predictSubredditOrientation`, {data}, { timeout: 0 })
      .then(res => {
        let getData = (res.data)[0];
        let leftResult = parseFloat(getData.left);
        let rightResult = parseFloat(getData.right);
        let orientationValue = "";
        
        if(leftResult > rightResult){
          orientationValue = "left";
        }else{
          orientationValue = "right";
        }

        setOrientation(orientationValue);
        
        let prediction = [data.subreddit, (leftResult*10).toFixed(2), (rightResult*10).toFixed(2), orientationValue];
        setPredictions(currPredictions => [...currPredictions, prediction]);
        setLoading(false);
      }).catch( err => {
        console.log(err);
        setError(true);
        setLoading(false);
      });
  }



  return (
      <div className="predictor-div-subreddit">
        <h1 className="prediction-text">Subreddit Political Position Predictor.</h1>
        <h3 className="prediction-text">{error ? "Failed to predict subreddit, try again." : (orientation ? "This subreddit is " + orientation + " wing" : "")}</h3>
        <form onSubmit={handleSubmit}>
            <input disabled={loading} placeholder="Enter Subreddit" type="text" id="subreddit" value={subreddit} onChange={e => (e.target.value.length < 50) && setSubreddit(e.target.value)}></input>
            <input disabled={loading} type="submit" value="Predict Alignment"/>
        </form>
        <hr></hr>
        <h3>Prediction Classification Breakdown:</h3>
        <table className="prediction-table" style={{width:"50%"}}>
          <tbody>
          <tr>
            <th>Subreddit</th>
            <th>Left Score</th>
            <th>Right Score</th>
            <th>Overall Result</th>
          </tr>
          {predictions.map((result, index) => (
            <tr key={index}>
              <td>{result[0]}</td>
              <td>{result[1]}</td>
              <td>{result[2]}</td>
              <td style={ result[3] === "right" ? {color:"red"} : {color:"blue"}}>{result[3]}</td>
            </tr>
          ))}
          </tbody>
        </table>
      </div>
  )
}

export default PredictorSubreddit