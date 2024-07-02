import React, { Component } from "react";
import * as venn from "venn.js";
import axios from 'axios';
import VennDiagram from '../components/VennDiagram'

//Dummy data to populate initial venn
var sets = [
  {
      "size": 3,
      "sets": [
          "conservative",
          "conspiracy"
      ]
  },
  {
      "size": 3,
      "sets": [
          "conspiracy"
      ]
  },
  {
      "size": 4,
      "sets": [
          "conservative",
          "worldnews"
      ]
  },
  {
      "size": 4,
      "sets": [
          "worldnews"
      ]
  },
  {
      "size": 7,
      "sets": [
          "conservative",
          "Libertarian"
      ]
  },
  {
      "size": 7,
      "sets": [
          "Libertarian"
      ]
  },
  {
      "size": 9,
      "sets": [
          "conservative",
          "news"
      ]
  },
  {
      "size": 9,
      "sets": [
          "news"
      ]
  },
  {
      "size": 11,
      "sets": [
          "conservative",
          "politics"
      ]
  },
  {
      "size": 11,
      "sets": [
          "politics"
      ]
  },
  {
      "size": 34,
      "sets": [
          "conservative"
      ]
  }
]

const backendUrl = process.env.REACT_APP_BACKEND_URL;

class AnalyseText extends Component {
  constructor(props) {
    super(props);
    this.chartView = React.createRef();
    this.state = {
      value: '',
      title: 'Conservative',
      isLoaded: true,
      chartData: sets
    }
    this.handleChange = this.handleChange.bind(this)
    this.handleSubmit = this.handleSubmit.bind(this)
    this.pullData = this.pullData.bind(this)
  }

  chart = venn.VennDiagram();

  pullData(subreddit){
    if(this.state.isLoaded === true){
      this.props.handleProgress(0);
      if(subreddit.toLowerCase() !== this.state.title.toLowerCase()){
        var data = {
          'subreddit': subreddit,
          'size':'15'
        }
        
        this.setState({
          value: this.state.value,
          title: 'Generating r/' + data.subreddit,
          isLoaded: false,
          chartData: this.state.chartData});
        axios.defaults.headers.post['Content-Type'] ='application/json;charset=utf-8';
        axios.defaults.headers.post['Access-Control-Allow-Origin'] = '*';
        axios.post(`${backendUrl}/pullSubreddits`, {data}, { timeout: 0 })
          .then(res => {
            this.props.handleProgress(100);
            let getData = res.data;
            this.setState({
              value: '',
              title: data.subreddit,
              isLoaded: true,
              chartData: getData});
          }).catch( err => {
            console.log(err);
          });
      }
    }
  }

  handleChange(event) {
    if(event.target.value.length < 50){
      let val = event.target.value;
      this.setState(prevState => {
        let value = Object.assign({}, prevState.value);
        value = val
        return { value }
      })
    }
  }

  handleSubmit(event){
    event.preventDefault()
    this.pullData(this.state.value)
  }

  render() {
    return (
      <div>
        <h1>{this.state.isLoaded ? "r/"+this.state.title : this.state.title}</h1>
        <VennDiagram vennData={this.state.chartData} loaded={this.state.isLoaded} searchVenn={this.pullData} current={this.state.title}/>
        <form onSubmit={this.handleSubmit} className="venn-diagram-form">
          <input disabled={!this.state.isLoaded} placeholder="Enter Subreddit" className="venn-input" type="text" name="name" value={this.state.value} onChange={this.handleChange}/>
          <br/>
          <input disabled={!this.state.isLoaded} type="submit" value="Submit" />
        </form>
      </div>
    );
  }
}

export default AnalyseText;
