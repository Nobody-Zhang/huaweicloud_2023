<template>
    <div class="header">
        <div class="logo" @click="goHome">
            <img src="@/assets/images/team_logo.png" alt="Logo" />
        </div>
        <div class="name">
          {{ title }}
        </div>
        <div class="menu">
            <BGM></BGM>
            <MenuBar></MenuBar>
        </div>
    </div>
</template>

<script>
import MenuBar from "../common/MenuBar/MenuBar.vue"
import BGM from "../common/BGM/BackgroundMusic.vue"

export default {
    name: "HeaderComp",
    components: {
        MenuBar,
        BGM,
    },
    props: {
      title: String,
    },
    methods:{
      goHome(){
        if (this.$route.path === "/") return;
        this.$router.push({path: "/"});
      }
    }
}
</script>

<style lang="scss">
@import "../../assets/scss/index.scss";
.header {
  font-family: $default_font_family;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: $light_yellow;
  color: black;
  height: 60px;
}

.logo img {
  height: 40px;
}

.name {
  // 在中间
  font-size: 22px;
  font-weight: 550;
}

.menu {
  // 在右边,向右对齐
  display: flex;
  float: right;
  gap: 10px;
}
</style>