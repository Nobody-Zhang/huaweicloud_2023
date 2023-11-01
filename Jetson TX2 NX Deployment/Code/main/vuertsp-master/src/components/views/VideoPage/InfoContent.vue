<template>
  <div class="info-content">
    <div class="info-left">
        <div class="info-top">
            <div class="info-row">
                <span class="info-label">Gender:</span>
                <span class="info-value">{{ gender }}</span>
            </div>
        </div>
        <div class="info-bottom">
            <div class="info-row">
                <span class="info-label">Time:</span>
                <span class="info-value">{{ time }}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Frame Rate:</span>
                <span class="info-value">{{ frameRate }}</span>
            </div>
        </div>
    </div>
    <div class="info-right">
      <div class="info-row">
        <span class="info-label">Cloud:</span>
        <span class="info-value" :class="{ 'online': isCloudOnline, 'offline': !isCloudOnline }">
          {{ isCloudOnline ? 'Online' : 'Offline' }}
        </span>
      </div>
      <div class="info-row">
        <span class="info-label">Network:</span>
        <span class="info-value" :class="{ 'online': isNetworkOnline, 'offline': !isNetworkOnline }">
          {{ isNetworkOnline ? 'Online' : 'Offline' }}
        </span>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'InfoContent',
  data(){
    return {
      gender: '男',
      time: this.getCurrentTime(),
      frameRate: '10',
      isCloudOnline: true,
      isNetworkOnline: true,
    }
  },
  mounted() {
    setInterval(() => {
      this.time = this.getCurrentTime();
    }, 1000);
  },
  methods: {
    getCurrentTime() {
      const date = new Date();
      const year = date.getFullYear();
      const month = date.getMonth() + 1;
      const day = date.getDate();
      const hour = date.getHours();
      const minute = date.getMinutes();
      const second = date.getSeconds();
      return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
    }
  },
  // 更改props的值

};
</script>

<style lang="scss" scoped>
@import "../../../assets/scss/index.scss";
.info-content {
  display: flex;
  flex-direction: row;
  justify-content: space-between;
  align-items: center;
  color: black;
  padding: 20px;
  border-radius: 5px;
  line-height: 36px;
  
  @media only screen and (max-width: $lg_window_size){
    padding: 8px;
  }
}

.info-left{
    display: flex;
    flex-direction: column;
}
.info-right {
    display: flex;
    flex-direction: row;
}
.info-top {
  flex: 1;
}

.info-bottom {
  flex: 1;
  text-align: left;
  display: flex;
  flex-direction: row;
  @media only screen and (max-width: $lg_window_size){
    justify-content: space-between;
  }
}

.info-row {
  margin: 4px 20px;
  @media only screen and (max-width: $lg_window_size){
    margin: 4px 6px;
  }
}

.info-label {
  font-weight: bold;
  margin-right: 5px;
}

.info-value {
  font-weight: normal;
}

.online {
  color: green;
}

.offline {
  color: red;
}
</style>
