<template>
    <div class="scroll-reveal professional-skills">
      <div class="professional-skills__header">
        <div class="professional-skills__icon">
          <img
            loading="lazy"
            src="../assets/professional-skills.svg"
            alt="Professional Skills Icon"
          />
        </div>
        <div class="professional-skills__title">
          <h2>Professional Skills</h2>
          <p>
            Key competencies essential for navigating and excelling in various
            professional contexts.
          </p>
        </div>
      </div>
      <div class="professional-skills__content">
        <SkillSection
          title="Frontend Frameworks & Tools"
          :skills="frontendSkills"
        />
        <SkillSection
          title="Backend Frameworks & Tools"
          :skills="backendSkills"
        />
      </div>
    </div>
  </template>
  
  <script>
  import SkillSection from './SkillSection.vue';
  
  export default {
    name: 'ProfessionalSkills',
    components: {
      SkillSection,
    },
    data() {
      return {
        frontendSkills: [
          'Vue.js', 'Nuxt.js', 'Vuex', 'SEO', 'Google Tag Manager',
          'React', 'Vuetify', 'Tailwind CSS', 'Pinia', 'Typescript',
          'Next.js', 'Javascript', 'HTML5', 'CSS3', 'SCSS',
          'SASS', 'CI-CD', 'AWS', 'Azure Cloud Services', 'Kubernetes',
          'Docker', 'Git', 'Jira'
        ],
        backendSkills: [
          'Java', 'Spring Boot', 'Python', 'Django', 'Flask',
          'Node.js', 'Numpy', 'Pandas', 'SQL Server', 'MySQL',
          'SQLite3', 'MongoDB'
        ]
      };
    }
  };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .professional-skills {
    margin-top: 1rem;
    background-color: #ffffff;
    border-radius: 1rem;
    border: 1px solid #e5e7eb;
    overflow: hidden;
  
    &__header {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      padding: 1.5rem;
      border-bottom: 1px solid #e5e7eb;
    }
  
    &__icon {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 3.25rem;
      height: 3.25rem;
      background-color: #f3f4f6;
      border-radius: 0.75rem;
      border: 1px solid #e5e7eb;
  
      img {
        width: 1.5rem;
        height: 1.5rem;
        object-fit: contain;
      }
    }
  
    &__title {
      margin-top: 1.5rem;
      max-width: 25rem;
  
      h2 {
        font-size: 1.25rem;
        font-weight: 500;
        line-height: 1.2;
        color: #111827;
      }
  
      p {
        margin-top: 0.75rem;
        font-size: 0.875rem;
        line-height: 1.25;
        color: #4b5563;
      }
    }
  
    &__content {
      padding: 2rem 1.5rem 2.5rem;
    }

    @media (min-width: 768px) {
      max-width: 720px;
      margin: 0px auto;
      margin-top: 24px;
    }

    @media (max-width: 768px) {
      max-width: 80%;
      margin: 0px auto;
      margin-top: 24px;
    }
  }
  
  @media (max-width: 768px) {
    .professional-skills {
      &__header {
        padding: 1rem;
      }
  
      &__content {
        padding: 1.5rem 1rem 2rem;
      }
    }
  }
  </style>