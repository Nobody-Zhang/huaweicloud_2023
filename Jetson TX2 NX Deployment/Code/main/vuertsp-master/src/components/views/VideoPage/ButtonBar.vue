<template>
    <div class="button-bar">
        <div
            v-for="(buttonEl, index) in buttonEls"
            :key="index"
            class="button-container"
            :class="isButtonColor[index] ? buttonEl.color : 'white'"
            @click="toggleButtonColor(index)"
        >
          <img :src="buttonEl.image" class="buttonImg" alt="button image"/>
        </div>
    </div>
</template>

<script>
import {closedEyes, headTurn, phoneWatch, yawnEmoji} from "../../../assets/images/button";
export default {
  name: 'ButtonBar',
  data() {
    return {
      buttonEls: [
        {
          type: 'closed-eyes',
          image: closedEyes,
          color: "pink_message",
        },
        {
          type: 'head-turn',
          image: headTurn,
          color: "light_blue_message",
        },
        {
          type: "phone-watch",
          image: phoneWatch,
          color: "grass_green_message",
        },
        {
          type: "yawn-emoji",
          image: yawnEmoji,
          color: "purple_message",
        }
        // Add more alert types and messages
      ],
      
      isButtonColor: [],
    };
  },
  
  methods: {
    toggleButtonColor(index) {
      this.$set(this.isButtonColor, index, !this.isButtonColor[index]);
    }
  }
};
</script>

<style lang="scss" scoped>
@import "../../../assets/scss/index.scss";
.button-bar{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.button-container{
    height: 0;
    padding: 50% 0%;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px black solid;
    border-radius: 8px;
    margin: 5px 0px;
    
    @media only screen and (max-width: $md_window_size){
      padding: 12% 5%;
      flex: 1;
    }
}

.button-container:hover{
    cursor: pointer;
}

.buttonImg{
    max-width: 40px;
    max-height: 40px;

    @media only screen and (min-width: $md_window_size) and (max-width: $lg_window_size) {    
      max-width: 32px;
      max-height: 32px;
    }
    @media only screen and (max-width: $md_window_size){
      max-width: 40px;
      max-height: 40px;
    }
}

.purple_message{
  background-color: $purple_message;
}
.pink_message{
  background-color: $pink_message;
}
.light_blue_message{
  background-color: $light_blue_message;
}
.grass_green_message{
  background-color: $grass_green_message;
}
</style>