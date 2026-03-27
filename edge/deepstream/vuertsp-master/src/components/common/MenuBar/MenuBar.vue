<template>
  <div class="menu-bar" >
    <router-link to="/about">关于</router-link>

  </div>
</template>

<script>
export default {
  name: 'MenuBar',
  data() {
    return {

    };
  },
};
</script>

<style lang="scss" scoped>
@import "../../../assets/scss/index.scss";
.menu-bar{
  position: fixed;
  top:23px;
  right: 60px;
  font-weight: 700;
  width: 70px;
  height: 30px;
  line-height: 30px;
  text-align: center;
  //border: 1px solid #000;
}

/* Add more styles for different alert types */
</style>
