<template>
    <div class="scroll-reveal language-card">
      <div class="language-card__header">
        <div class="language-card__icon">
          <img
            loading="lazy"
            src="../assets/globe.svg"
            class="language-card__icon-image"
            alt="Language icon"
          />
        </div>
        <div class="language-card__title-container">
          <h2 class="language-card__title">Languages</h2>
          <p class="language-card__description">
            Proficient in multiple languages, facilitating effective communication
            and collaboration in diverse settings.
          </p>
        </div>
      </div>
      <div class="language-card__content">
        <div class="language-item">
          <h3 class="language-item__name">English</h3>
          <p class="language-item__proficiency">
            Professional working proficiency
          </p>
        </div>
        <div class="language-card__divider"></div>
        <div class="language-item">
          <h3 class="language-item__name">Telugu</h3>
          <p class="language-item__proficiency">
            Native or bilingual proficiency
          </p>
        </div>
        <div class="language-card__divider"></div>
      </div>
    </div>
  </template>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .language-card {
    display: flex;
    flex-direction: column;
    margin-top: 0.75rem;
    width: 100%;
    background-color: $white;
    border-radius: 1rem;
    border: 1px solid $gray-200;
  
    &__header {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      padding: 1.5rem;
      width: 100%;
      border-bottom: 1px solid $gray-200;
    }
  
    &__icon {
      display: flex;
      gap: 0.625rem;
      align-items: center;
      padding: 0.875rem;
      border-radius: 0.75rem;
      border: 1px solid $gray-200;
      background-color: $gray-100;
      height: 3.25rem;
      width: 3.25rem;
  
      &-image {
        object-fit: contain;
        width: 1.5rem;
        aspect-ratio: 1 / 1;
      }
    }
  
    &__title-container {
      display: flex;
      flex-direction: column;
      margin-top: 1.5rem;
      font-weight: 500;
      color: $black;
      max-width: 25rem;
      width: 100%;
    }
  
    &__title {
      font-size: 1.25rem;
      line-height: 1.2;
      margin: 0;
      font-weight: 500;
    }
  
    &__description {
      margin-top: 0.75rem;
      font-size: 0.875rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &__content {
      display: flex;
      flex-direction: column;
      padding: 2rem 1.5rem 2.5rem;
      width: 100%;
      font-weight: 500;
      color: $black;
    }
  
    &__divider {
      display: flex;
      width: 100%;
      background-color: $gray-200;
      min-height: 1px;
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
  
  .language-item {
    display: flex;
    flex-direction: column;
    width: 100%;
  
    &__name {
      font-size: 1rem;
      line-height: 1.25;
      margin: 0;
      font-weight: 500;
    }
  
    &__proficiency {
      margin-top: 0.375rem;
      font-size: 0.875rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &:not(:first-child) {
      margin-top: 1rem;
    }
  }
  
  @media (max-width: 768px) {
    .language-card {
      &__header {
        padding: 1.25rem;
      }
  
      &__content {
        padding: 1.25rem;
      }
  
      &__title-container {
        max-width: 100%;
      }
    }
  }
  </style>