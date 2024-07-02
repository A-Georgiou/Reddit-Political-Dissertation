import React from "react"
import tune from '../static/music/whistlin-tune.mp3'
import '../styles/404.scss';

const NotFoundPage = () => {
  return (
    <div>
      <h1 style={{textAlign:"center",margin:"1em auto"}}>Page 404: Nothin' to see here. Carry on and have a nice day :)</h1>
      <audio controls autoPlay loop style={{display:"block",margin:"5em auto"}}>
        <source src={tune} type="audio/mpeg" />
      </audio>
      <a href="/"><img src="https://www.icegif.com/wp-content/uploads/cool-icegif-2.gif" alt="cool cat" title="Take me home" style={{display:"block",margin:"auto"}}/></a>
    </div>
  )
}

export default NotFoundPage
