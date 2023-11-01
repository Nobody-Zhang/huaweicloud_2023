<template>
    <div>
      <button @click="togglePlay" class="mu"><img :src="musicImg" class="musicImg"></button>
      <audio ref="audio" :src="musicSource" loop class="audio"></audio>
    </div>
  </template>
  
  <script>
  export default {
    name:"BGM",
    data() {
      return {
        isPlaying: false,
        musicImg: require('@/assets/images/button/music.png'),
        musicSource: require('@/assets/audio/NWGYU.wav')
      };
    },
    methods: {
      togglePlay() {
        const audioElement = this.$refs.audio;
        if (this.isPlaying) {
          audioElement.pause();
        } else {
          audioElement.play();
        }
        this.isPlaying = !this.isPlaying;
      }
    }
  };
  </script>
  <style lang="scss" scoped>
  .musicImg {
    background-color: transparent;
    width: 30px;
    height: 30px;
  }
  .audio{
    width: 0%;
    height: 0%;
  }
  .mu{
    background-color: transparent;
    border: none;
    border-radius: 20%;
    &:hover {
      opacity: 0.8;
      background-color: #9999;
    }
    &:active{
      background-color: darken($color: #9999, $amount: 20%);
    }
  }
  </style>