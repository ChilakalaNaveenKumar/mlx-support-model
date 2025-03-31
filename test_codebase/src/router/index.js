import { createRouter, createWebHistory } from 'vue-router';
// import BuilderPage from '../BuilderPage.vue';
import ResumeMaker from '@/components/resume-maker.vue';

const routes = [
  { path: '/resume-template', component: ResumeMaker},
  { path: '/', component: ResumeMaker},
  // { path: '/:slug(.*)', component: BuilderPage }, // Dynamic route for Builder pages
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
