import React, { useState } from 'react';
import axios from 'axios';

const backendUrl = process.env.REACT_APP_BACKEND_URL;

const Predictor = () => {
  const [comment, setComment] = useState('');
  const [orientation, setOrientation] = useState('');
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);
    const data = {
      comment: comment
    };

    axios.post(`${backendUrl}/predictOrientation`, { data }, { timeout: 0 })
      .then(res => {
        const getData = res.data[0];
        const leftResult = parseFloat(getData.left);
        const rightResult = parseFloat(getData.right);

        let orientationValue = '';

        if (leftResult >= 0.85) {
          orientationValue = 'confidently left';
        } else if (leftResult > 0.5) {
          orientationValue = 'potentially left';
        } else if (rightResult >= 0.85) {
          orientationValue = 'confidently right';
        } else {
          orientationValue = 'potentially right';
        }

        setOrientation(orientationValue);

        const prediction = [data.comment, (leftResult * 100).toFixed(2), (rightResult * 100).toFixed(2)];
        setPredictions(currPredictions => [...currPredictions, prediction]);
        setLoading(false);
      })
      .catch(err => {
        console.log(err);
        setLoading(false);
      });
  };

  return (
    <div className="predictor-div">
      <h1 className="prediction-text">Political Commentary Predictor.</h1>
      <h3 className="prediction-text">{orientation ? `This comment is ${orientation} wing` : ''}</h3>
      <form onSubmit={handleSubmit}>
        <textarea
          disabled={loading}
          type="text"
          id="comment"
          value={comment}
          onChange={e => e.target.value.length < 512 && setComment(e.target.value)}
          placeholder="Enter Political Comment"
          rows="4"
          cols="50"
        ></textarea>
        <br />
        <input type="submit" value="Predict Alignment" id="predict-form-submit" disabled={loading} />
      </form>
      <hr />
      <h3>Prediction Classification Breakdown:</h3>
      <table className="prediction-table" style={{ width: '50%' }}>
        <tbody>
          <tr>
            <th>Comment</th>
            <th>Left</th>
            <th>Right</th>
          </tr>
          {predictions.map((result, index) => (
            <tr key={index}>
              <td>{result[0]}</td>
              <td>{result[1]}%</td>
              <td>{result[2]}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default Predictor;