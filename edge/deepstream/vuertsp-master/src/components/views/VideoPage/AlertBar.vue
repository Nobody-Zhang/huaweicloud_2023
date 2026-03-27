<template>
  <div class="alert-bar">
    <div class="alert-count">
      
    </div>
    <div
      v-for="(alert, index) in alertlist"
      :key="index"
      class="alert"
    >
      <img :src="alert.image" class="alert-icon" />
      <span class = "alert-type" :class="alert.type">{{ alert.title }}:</span>
        {{ alert.message }}
    </div>
  </div>
</template>

<script>
export default {
  name: 'AlertBar',
  data() {
    return {
      alertlist: [
        {
          type: 'None',
          title: "None",
          message: 'No alert',
        },
      ],
      alerts: [
        {
          type: 'normal',
          title: "Normal",
          message: 'User is not showing any signs of drowsiness',
        },
        {
          type: 'closed-eyes',
          title: "Closed Eyes",
          message: 'User is closing their eyes for too long',
        },
        {
          type: 'yawn-emoji',
          title: "Yawning",
          message: 'User is yawning too much',
        },
        {
          type: 'phone',
          title: "Phone",
          message: 'User is using their phone',
        },
        {
          type: 'head-turn',
          title: "Head Turn",
          message: 'User is turning their head away',
        }
      ],
      
    };
  },
  mounted() {
    this.ws = new WebSocket('ws://10.21.184.155:7980');
    this.ws.onmessage = this.handleMessage;
  },
  methods: {
    handleMessage(event) {
      // 收到数字代表是几个alert
      const message = event.data;
      if (message >= 0 && message <= 4) {
        this.alertlist.push(this.alerts[message]);
        if (this.alertlist.length > 14) {
          this.alertlist.shift();
        }
      }
    }
  }

};
</script>

<style lang="scss" scoped>
@import "../../../assets/scss/index.scss";
.alert-bar {
  color: white;
  background-color: dimgray;
  opacity: 0.8;
  padding: 10px;
  font-family: $default_font_family;
}

.alert {
  display: inline-block;
  margin-bottom: 5px;
  padding: 8px;
  border-radius: 5px;
  font-size: 14px;
  line-height: 1.4;
  width: 90%;
}

.alert-type{
  font-weight: 550;
}

.normal {
  color: #9999;
}



.yawn-emoji {
  color: $purple_message;
}

.phone {
  color: $grass_green_message;
}

.closed-eyes {
  color: $pink_message;
}

.head-turn {
  color: $light_blue_message;
}

/* Add more styles for different alert types */
</style>
