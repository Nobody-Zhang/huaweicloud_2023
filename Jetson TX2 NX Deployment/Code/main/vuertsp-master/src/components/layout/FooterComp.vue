<template>
    <footer class="footer">
      <p>{{ teamName }} &copy; 2023</p>
    </footer>
</template>
  
<script>
  export default {
    name: "FooterComp",
    props: {
        teamName: String,
    },
  };
</script>
  
<style lang="scss" scoped>
@import "../../assets/scss/style.scss";
.footer {
    display: flex;
    justify-content: center;
    background-color: $yellow;
    color: black;
    text-align: center;
    padding: 10px;
}
</style>
  