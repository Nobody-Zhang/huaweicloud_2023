<template>
  <div class="video-container">
    <!-- <canvas ref="videoCanvas" class="video-content" :src="currentFrame"> -->
    <img ref="image" :src="currentFrame" alt="Video" class="video-content"/>
    <!-- </canvas> -->
    <div class="button-bar-container">
    <div class="button-bar">
      <button @click="startVideo" class="but"><img :src="playimg" class="buttonImg"/></button>
      <button @click="stopVideo" class="but"><img :src="stopimg" class="buttonImg"/></button>
      <button @click="saveVideo" class="but"><img :src="saveimg" class="buttonImg"/></button>
    </div>
    </div>
  </div>
</template>
  
  <script>
  // import JSMpeg from '../../../assets/server/jsmpeg.min.js';
  // import WebSocket from 'ws';
  export default {
    name: 'VideoContent',
    data() {
      return {
        ws: null,
        currentFrame: '',
        isPlaying: false,
          playimg: require('../../../assets/images/button/play.png'),
          saveimg: require('../../../assets/images/button/save.png'),
          stopimg: require('../../../assets/images/button/stop.png'),
      };
    },
    mounted() {
      this.ws = new WebSocket('ws://10.21.184.155:7979');
      this.ws.onmessage = this.handleMessage;
  },
  methods: {
    startVideo() {
      if (!this.isPlaying) {
        this.isPlaying = true;
        this.ws.send('start'); 
      }
    },
    stopVideo() {
      if (this.isPlaying) {
        this.isPlaying = false;
        this.ws.send('stop');
      }
    },
    saveVideo() {
      this.ws.send('save');
    },
    handleMessage(event) {
      const message = event.data;
      const blob = new Blob([message], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);

      this.currentFrame = url
    }
  }
};
    
    /*
    mounted() {
      this.player = new JSMpeg.Player('ws://localhost:9999', {
        canvas: this.$refs.canvas,
      });
    },
    beforeDestroy() {
      if (this.player) {
        this.player.destroy();
      }
    },
    */
  // };
  </script>
  
  <style lang="scss" scoped>
  @import "../../../assets/scss/index.scss";
  .video-container {
    width: 100%;
    height: 100%;
    position: relative;
    background-color: black;
  }
  .video-content {    
    width: 100%;
    height: 90%;
  }

  .but{
    opacity: 0.8;
    background-color: #ffff;
    border-radius: 5px;
    border: #ffff;
    &:hover {
      background-color: rgba(#ffff, 0.7);
    }
    &:active {
      // active时，背景色变浅，且与focus，hover做区分
      background-color: darken($color: #ffff, $amount: 10%)
    }
  }
  .button-bar-container{
      width:100%;
      opacity: 0.7;
      background-color: #9999;
      display: flex;
      position: absolute;
      left: 0%;
      bottom: 0%;

  }
    .button-bar{
      width: 80px;
      display:flex;
      border-radius: 5px;
      flex-direction: row;
      justify-content: space-between;
    }

  .buttonImg{
    // background-color: #ffff;
    width: 100%;
    height: 100%;
    max-width: 15px;
    max-height: 15px;
  }

  </style>
  