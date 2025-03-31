<template>
    <div class="skill-section">
      <h3 class="skill-section__title">{{ title }}</h3>
      <ul class="skill-section__list">
        <li v-for="skill in skills" :key="skill" class="skill-section__item">
          <img
            loading="lazy"
            src="../assets/checkmark.svg"
            alt="Skill Icon"
          />
          <span>{{ skill }}</span>
        </li>
      </ul>
    </div>
  </template>
  
  <script>
  export default {
    name: 'SkillSection',
    props: {
      title: {
        type: String,
        required: true
      },
      skills: {
        type: Array,
        required: true
      }
    }
  };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .skill-section {
    &:not(:first-child) {
      margin-top: 2.5rem;
    }
  
    &__title {
      font-size: 1rem;
      font-weight: 500;
      line-height: 1.25;
      color: #111827;
    }
  
    &__list {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 1rem;
      padding: 0px;
    }
  
    &__item {
      display: flex;
      align-items: center;
      gap: 0.375rem;
      padding: 0.375rem 0.625rem 0.375rem 0.75rem;
      background-color: #ffffff;
      border-radius: 0.5rem;
      border: 1px solid #e5e7eb;
      font-size: 0.75rem;
      font-weight: 500;
      line-height: 1.25;
      color: #111827;
      white-space: nowrap;
  
      img {
        width: 0.75rem;
        height: 0.75rem;
        object-fit: contain;
      }
    }
  }
  
  @media (max-width: 768px) {
    .skill-section {
      &__list {
        gap: 0.375rem;
      }
  
      &__item {
        padding: 0.25rem 0.5rem 0.25rem 0.625rem;
        font-size: 0.6875rem;
      }
    }
  }
  </style>