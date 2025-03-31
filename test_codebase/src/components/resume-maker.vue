<template>
  <NavigationHeader />
  <ProfileHeader id="home" />
  <ProfileActions id="profile" />
  <ProfileCard />
  <WorkedWith />
  <WorkExperience id="experience" />
  <EducationComponent id="education" />
  <ProfessionalSkills id="skills" />
  <LanguageCard />
  <HonorsAndAwards />
  <FloatingButton />
  <ContactSection id="contactme" />
  <Footer />
</template>

<script setup>
import { onMounted } from 'vue';
import NavigationHeader from "./NavigationHeader.vue";
import ProfileHeader from "./ProfileHeader.vue";
import ProfileActions from "./ProfileActions.vue";
import ProfileCard from "./ProfileCard.vue";
import WorkedWith from "./WorkedWith.vue";
import WorkExperience from "./WorkExperience.vue";
import EducationComponent from "./EducationComponent.vue";
import ProfessionalSkills from "./ProfessionalSkills.vue";
import LanguageCard from "./LanguageCard.vue";
import HonorsAndAwards from "./HonorsAndAwards.vue";
import FloatingButton from "./FloatingButton.vue";
import ContactSection from "./ContactForm/ContactSection.vue";
import Footer from "./Footer.vue";

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.scroll-reveal').forEach((el) => {
    observer.observe(el);
  });
});
</script>

<style>
.scroll-reveal {
  opacity: 0;
  transform: translateY(20px);
  transition: opacity 0.5s ease, transform 0.5s ease;
}

.scroll-reveal.visible {
  opacity: 1;
  transform: translateY(0);
}
</style>