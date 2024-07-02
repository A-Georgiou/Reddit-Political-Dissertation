import React, { useState, useRef } from 'react';
import sketch from '../scripts/sketches/word-frequency.js';
import Introduction from '../components/introduction.jsx';
import { graphql } from "gatsby";
import AnalyseText from '../components/AnalyseText.jsx';
import Predictor from '../components/Predictor.jsx';
import PredictorSubreddit from '../components/PredictorSubreddit.jsx';
import LoadingBar from 'react-top-loading-bar';
import '../styles/main.scss';
import loadable from "@loadable/component";

const LoadableP5Wrapper = loadable(() => import('@p5-wrapper/react').then(module => module.ReactP5Wrapper));

export const query = graphql`
  query {
    allWordFreqNewCsv {
        nodes {
          word
          orientation
        }
      }
  }
`;

const IndexPage = ({data}) => {
  const dataNodes = data.allWordFreqNewCsv.nodes;
  const [numWords, setNumWords] = useState(20);
  const ref = useRef(null)
  const handleSelect = (e) => setNumWords(e.target.value);

  const handleProgress = (progression) =>  {
    if(progression===100){
      ref.current.complete();
    }else{
      ref.current.continuousStart(0, 4000);
    }
  };

  return (
    <React.Fragment>
      <div className="note-title"><p><b>Note: </b>This project was completed as part of my Computer Science Bsc dissertation. The Backend API is no longer hosted due to long-term AWS EC2 costs.</p></div>
      <title>Reddit Analysis Project</title>
      <Introduction/>
      <LoadingBar  ref={ref} shadow={true} color='#719ECE' height={10}/>
      <div className="venn-diagram">
        <AnalyseText handleProgress={handleProgress}/>
      </div>
      <div className="word-frequency">
        <h1>Word Frequency List (<span style={{color:"blue"}}>Left Wing</span> / <span style={{color:"red"}}>Right Wing</span>)</h1>
        <LoadableP5Wrapper sketch={sketch} data={dataNodes} numWords={numWords}/>
        <div className="word-display-div">
        <label>Number of words to display: </label>
        <select id="num-words" name="Num-words" onChange={e => handleSelect(e)}>
            <option value="20">20</option>
            <option value="50">50</option>
            <option value="150">150</option>
            <option value="250">250</option>
            <option value="500">500</option>
        </select>
        </div>
      </div>
      <Predictor/>
      <PredictorSubreddit/>
    </React.Fragment>
  )
}

export default IndexPage
