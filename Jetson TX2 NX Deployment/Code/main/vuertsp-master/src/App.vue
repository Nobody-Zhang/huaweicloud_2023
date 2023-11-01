<template>
  <div id="app">

    <canvas id="bgcanvas" class="bgcanvas"></canvas>
    <HeaderComp title="华为云人工智能疲劳驾驶检测"></HeaderComp>
    <router-view></router-view>
  </div>
</template>

<script>
//import FooterComp from "./components/layout/FooterComp.vue";
import HeaderComp from "./components/layout/HeaderComp.vue";
export default {
  name: 'App',
  components: {
    HeaderComp,
    // FooterComp,
  },
  mounted() {


    class Circle {
      constructor(x, y) {
        this.x = x;
        this.y = y;
        this.r = Math.random() * 10;
        this._mx = Math.random();
        this._my = Math.random();

      }
      drawCircle(ctx) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.r, 0, 360)
        ctx.closePath();
        ctx.fillStyle = 'rgba(204, 204, 204, 0.3)';
        ctx.fill();
      }

      drawLine(ctx, _circle) {
        let dx = this.x - _circle.x;
        let dy = this.y - _circle.y;
        let d = Math.sqrt(dx * dx + dy * dy)
        if (d < 150) {
          ctx.beginPath();
          ctx.moveTo(this.x, this.y); //起始点
          ctx.lineTo(_circle.x, _circle.y); //终点
          ctx.closePath();
          ctx.strokeStyle = 'rgba(204, 204, 204, 0.3)';
          ctx.stroke();
        }
      }

      move(w, h) {
        this._mx = (this.x < w && this.x > 0) ? this._mx : (-this._mx);
        this._my = (this.y < h && this.y > 0) ? this._my : (-this._my);
        this.x += this._mx / 2;
        this.y += this._my / 2;
      }
    }


    class currentCirle extends Circle {
      constructor(x, y) {
        super(x, y)
      }

      drawCircle(ctx) {
        ctx.beginPath();
        this.r = 8;
        ctx.arc(this.x, this.y, this.r, 0, 360);
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 77, 54, 0.6)'
        ctx.fill();

      }
    }

    window.requestAnimationFrame = window.requestAnimationFrame || window.mozRequestAnimationFrame || window.webkitRequestAnimationFrame || window.msRequestAnimationFrame;

    let canvas = document.getElementById('bgcanvas');
    let ctx = canvas.getContext('2d');
    let w = canvas.width = canvas.offsetWidth;
    let h = canvas.height = canvas.offsetHeight;
    let circles = [];
    let current_circle = new currentCirle(0, 0)

    let draw = function() {
      ctx.clearRect(0, 0, w, h);
      for (let i = 0; i < circles.length; i++) {
        circles[i].move(w, h);
        circles[i].drawCircle(ctx);
        for (let j = i + 1; j < circles.length; j++) {
          circles[i].drawLine(ctx, circles[j])
        }
      }
      if (current_circle.x) {
        current_circle.drawCircle(ctx);
        for (var k = 1; k < circles.length; k++) {
          current_circle.drawLine(ctx, circles[k])
        }
      }
      requestAnimationFrame(draw)
    }

    let init = function(num) {
      for (var i = 0; i < num; i++) {
        circles.push(new Circle(Math.random() * w, Math.random() * h));
      }
      draw();
    }

    window.addEventListener('load', init(60));

    window.onmousemove = function(e) {
      e = e || window.event;
      current_circle.x = e.clientX;
      current_circle.y = e.clientY;
    }

    window.onmouseout = function() {
      current_circle.x = null;
      current_circle.y = null;
    }
  }
};
</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  border: none;
  position: relative;
}




.bgcanvas {
/*//opacity: 0.5;*/

  position:absolute;
  z-index: -1;

  display: block;
  width: 100%;
  height: 100%;
}

/* Demo Buttons Style */
.codrops-demos {
  font-size: 0.8em;
  text-align:center;
  position:absolute;
  z-index:99;
  width:96%;
}

.codrops-demos a {
  display: inline-block;
  margin: 0.35em 0.1em;
  padding: 0.5em 1.2em;
  outline: none;
  text-decoration: none;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 700;
  border-radius: 2px;
  font-size: 110%;
  border: 2px solid transparent;
  color:#fff;
}

.codrops-demos a:hover,
.codrops-demos a.current-demo {
  border-color: #383a3c;
}

a {
  color: inherit;
  text-decoration: none;
}
</style>
