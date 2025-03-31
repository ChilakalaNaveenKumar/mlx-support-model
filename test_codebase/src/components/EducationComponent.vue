<template>
    <div class="scroll-reveal education-component">
      <div class="education-header">
        <div class="education-icon">
          <img loading="lazy" src="../assets/education.svg" alt="Education icon" />
        </div>
        <div class="education-title">
          <h2>Education</h2>
          <p>Degrees and certifications in the specialized fields of Information Technology and Computer & Info Systems.</p>
        </div>
      </div>
      <div class="education-content">
        <EducationItem
          school="Auburn University at Montgomery"
          degree="Master's degree"
          field="Computer & Info Systems"
          startDate="May 2023"
          endDate="Dec 2024"
        />
        <div class="education-divider"></div>
        <EducationItem
          school="Gokaraju Rangaraju Institute Of Engineering & Technology"
          degree="Bachelor's degree"
          field="Information Technology"
          startDate="May 2013"
          endDate="Jun 2017"
        />
      </div>
    </div>
  </template>
  
  <script>
  import EducationItem from './EducationItem.vue';
  
  export default {
    name: 'EducationComponent',
    components: {
      EducationItem,
    },
  };
  </script>
  
  <style lang="scss" scoped>
   @use '../assets/main.scss' as *;
  .education-component {
    display: flex;
    flex-direction: column;
    margin-top: 1rem;
    width: 100%;
    background-color: $white;
    border-radius: 1rem;
    border: 1px solid $gray-200;
  
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
  
  .education-header {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 1.5rem;
    width: 100%;
    border-bottom: 1px solid $gray-200;
  
    @media (max-width: 768px) {
      padding: 1.25rem;
    }
  }
  
  .education-icon {
    display: flex;
    gap: 0.625rem;
    align-items: center;
    padding: 0.875rem;
    border-radius: 0.75rem;
    border: 1px solid $gray-200;
    background-color: $gray-100;
    height: 52px;
    width: 52px;
  
    img {
      object-fit: contain;
      width: 1.5rem;
      aspect-ratio: 1;
    }
  }
  
  .education-title {
    display: flex;
    flex-direction: column;
    margin-top: 1.5rem;
    font-weight: 500;
    color: $black;
    max-width: 400px;
    width: 100%;
  
    h2 {
      font-size: 1.25rem;
      line-height: 1.25;
      margin: 0px;
      font-weight: 500;
    }
  
    p {
      margin-top: 0.75rem;
      font-size: 0.875rem;
      line-height: 1.25;
      color: $gray-500;
    }
  }
  
  .education-content {
    display: flex;
    flex-direction: column;
    padding: 2rem 1.5rem 2.5rem;
    width: 100%;
    font-weight: 500;
    color: $black;
  
    @media (max-width: 768px) {
      padding: 1.25rem;
    }
  }
  
  .education-divider {
    display: flex;
    margin-top: 1.5rem;
    width: 100%;
    background-color: $gray-200;
    min-height: 1px;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  </style>