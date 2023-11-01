import Vue from 'vue';
import VueRouter from 'vue-router';
import VideoPage from "../components/views/VideoPage/VideoPage.vue"
import AboutIndex from "../components/views/About/Index.vue"

Vue.use(VueRouter);

const routes = [
    {
      path: '/',
      name: 'VideoPage',
      component: VideoPage,
    },

    {
        path: '/about',
        name: 'AboutIndex',
        component: AboutIndex,
    },
];

const router = new VueRouter({
    routes,
});

export default router;
