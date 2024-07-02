const parseCSVString = (csvString) => {
  // Regular expression to match the pattern "('word', number)"
  const regex = /\('([^']+)',\s*(\d+)\)/;
  const match = csvString.match(regex);
  
  if (match) {
    const word = match[1];        // Extract the word
    const frequency = parseInt(match[2], 10);  // Extract the frequency and convert to integer
    return { word, frequency };
  }
  
  return null; // Return null if the string doesn't match the pattern
};

export default function sketch(p) {
  const isBrowser = typeof window !== "undefined"
    
    // ~~~~~~ Initialize variables ~~~~~~~~~
    var data = [];
    var balls = [];
    let numBalls = 20;
    let spring = 0.0005;
    let friction = -1;
    let width = 800;
    let height = 600;
    if(isBrowser){
      width = p.windowWidth;
      height = (p.windowHeight)*(90/100);
    }

    function scaleVal(val) {
        if(val > 8000){
            return 1;
        }else if(val > 6000){
            return 0.95;
        }else if(val > 4000){
            return 0.9;
        }else if(val > 1000){
            return 0.8;
        }else if(val > 500){
            return 0.5;
        }else{
            return 0.4;
        }
     }

    // ~~~~~~ Props Import Handler ~~~~~~
    p.updateWithProps = (props) => {
      balls = [];
      if (props.numWords) {
        numBalls = props.numWords;
      }
  
      if (props.data) {
        data = props.data;
        for (let i = 0; i < numBalls; i++) {
          const datum = data[i];
          const wordFreq = datum.word;
          
          const {word, frequency} = parseCSVString(wordFreq);
          const scaleAmount = scaleVal(frequency);
  
          let color;
          if (datum.orientation === "0") {
            color = p.color(83, 165, 252, 255 * scaleAmount);
          } else {
            color = p.color(250, 67, 30, 255 * scaleAmount);
          }
          balls[i] = new Ball(
            p.sqrt((width * height * frequency) / 500000),
            i,
            balls,
            color,
            `${word}: ${frequency}`
          );
        }
        p.redraw(); // Explicitly call redraw after updating props
      }
    };

    // ~~~~~~ Setup ~~~~~~
    p.setup = () => {
        p.createCanvas(width,height);
        p.noStroke();
    }

    // ~~~~~~ Draw ~~~~~~
    p.draw = () => {
        p.background(245);
        balls.forEach(ball => {
            ball.collide();
            ball.move();
            ball.display();
        });
        checkChangeSize();
    }

    function checkChangeSize(){
      if(isBrowser){
        let newHeight = (p.windowHeight)*(90/100)
        let newWidth = p.windowWidth
        if(width !== newWidth || height !== newHeight){
            width = newWidth;
            height = newHeight;
            p.resizeCanvas(width, height);
        }
      }
    }

    // ~~~~~~ Ball Class ~~~~~~~~~~~~
    class Ball {
        constructor(din, idin, oin, color, text) {
          this.vx = 0.1;
          this.vy = 0.1;
          this.position = new p.createVector(p.random(0+(din/2), width-(din/2)), p.random(0+(din/2), height-(din/2)));
          this.velocity = new p.createVector(p.random(0.1, 0.2), p.random(0.1,0.2));
          this.diameter = din;
          this.id = idin;
          this.color = color;
          this.text = text;
          this.others = oin;
        }
      
        collide() {
          for (let i = this.id + 1; i < numBalls; i++) {
            let dx = this.others[i].position.x - this.position.x;
            let dy = this.others[i].position.y - this.position.y;
            let distance = p.sqrt(dx * dx + dy * dy);
            let minDist = this.others[i].diameter / 2 + this.diameter / 2;
            if (distance < minDist) {
              let angle = p.atan2(dy, dx);
              let targetX = this.position.x + p.cos(angle) * minDist;
              let targetY = this.position.y + p.sin(angle) * minDist;
              let ax = (targetX - this.others[i].position.x) * spring;
              let ay = (targetY - this.others[i].position.y) * spring;
              this.velocity.x -= ax;
              this.velocity.y -= ay;
              this.others[i].velocity.x += ax;
              this.others[i].velocity.y += ay;
            }
          }
        }
      
        move() {
          this.position.add(this.velocity);

          if ((this.position.x + this.diameter / 2 > width) || (this.position.x - (this.diameter / 2) <= 0)) {
            this.velocity.x = this.velocity.x * friction;
          }
          if ((this.position.y + this.diameter / 2 > height) || (this.position.y - (this.diameter / 2) < 0)) {
            this.velocity.y = this.velocity.y * friction;
          }
        }
      
        display() {
            p.fill(this.color);
            p.ellipse(this.position.x, this.position.y, this.diameter, this.diameter);
            p.fill(p.color(25, 25, 25, 200));
            p.textAlign(p.CENTER);
            p.textSize(this.diameter/5);
            p.text(this.text, this.position.x, this.position.y)
        }
      }
  }
  