<template>
    <div class="video-page">
        <div class="video-stuff">
            <div class="video-section">
                <ButtonBar class="button-sttuff"/>
                <div class="video-content-container">
                    <VideoContent />
                </div>
            </div>
            <div class="info-stuff">
                <InfoContent />
            </div>
        </div>
        <div class="alert-stuff">
            <AlertBar />
        </div>
        
    </div>
</template>
  
<script>
import AlertBar from './AlertBar.vue';
import VideoContent from './VideoContent.vue';
import InfoContent from './InfoContent.vue';
import ButtonBar from "./ButtonBar.vue";
  
export default {
    name: 'VideoPage',
    components: {
        AlertBar,
        VideoContent,
        InfoContent,
        ButtonBar,
    },
};
</script>
  
<style lang="scss" scoped>
@import "../../../assets/scss/index.scss";

.video-page{
    display: flex;
    flex-direction: row;
    justify-content: center;
    margin-top: 20px;
    width: 100%;
    height: 100%;
    
    @media only screen and (max-width: $lg_window_size){
        flex-direction: column;
    }
}

.video-stuff{
    display: flex;
    flex-direction: column;
    margin-right: 20px;
    height: 60%;
    width: 60%;

    @media only screen and (max-width: $lg_window_size){
        margin: 0px auto 20px auto;
        height: 98%;
        width: 98%;
    }
}


.video-section{
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    height: 70%;
    width: 100%;
    @media only screen and (max-width: $md_window_size){
        flex-direction: column-reverse;
    }

    .button-sttuff{
        max-width: 8%;
        flex: 1;
        border: 1px solid #000;
        border-radius: 4px;
        padding: 8px;

        display: flex;
        flex-direction: column;
        justify-content: space-between;
        
        user-select: none;

        @media only screen and (max-width: $md_window_size){
            flex-direction: row;
            max-width: 98%;
            margin: 0px 1% 0px 1%;
        }
    }

    .video-content-container{
        max-width: calc(92% - 36px); // margin 20px
        flex: 1;
        border: 1px solid #000;
        border-radius: 4px;
        @media only screen and (max-width: $md_window_size){
            max-width: 98%;
            margin: 0px 1% 20px 1%;
        }
    }
}

.info-stuff{
    width:100%;
    height: calc(30% - 20px);
    margin-top: 20px;
    border: 1px solid #000;
    border-radius: 4px;
    
    @media only screen and (max-width: $md_window_size){
        max-width: 98%;
        margin: 20px 1% 0px 1%;
    }
}

.alert-stuff{
    flex: 1;
    opacity: 0.9;
    max-width: 20%;
    background-color: dimgray;
    border-radius: 4px;
    border: 1px solid #000;

    
    @media only screen and (max-width: $lg_window_size){
        max-width: 98%;
        margin: 0px 1%;
    }
}

</style>
  