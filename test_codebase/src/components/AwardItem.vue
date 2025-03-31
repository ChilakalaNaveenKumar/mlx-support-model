<template>
    <div class="award-item">
      <div class="award-item__content">
        <div class="award-item__header">
          <h3 class="award-item__title">{{ title }}</h3>
          <div class="award-item__meta">
            <span>Issued by {{ issuer }}</span>
            <span>{{ role }}</span>
          </div>
        </div>
        <p class="award-item__description">{{ description }}</p>
      </div>
      <div class="award-item__date">
        <img
          loading="lazy"
          src="../assets/calendar.svg"
          alt="Calendar icon"
          class="award-item__date-icon"
        />
        <span>{{ date }}</span>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'AwardItem',
    props: {
      title: {
        type: String,
        required: true,
      },
      issuer: {
        type: String,
        required: true,
      },
      role: {
        type: String,
        required: true,
      },
      description: {
        type: String,
        required: true,
      },
      date: {
        type: String,
        required: true,
      },
    },
  };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .award-item {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-width: 15rem;
  
    &__content {
      display: flex;
      flex-direction: column;
      width: 100%;
    }
  
    &__header {
      display: flex;
      flex-direction: column;
      width: 100%;
      min-height: 2.625rem;
    }
  
    &__title {
      font-size: 1rem;
      line-height: 1.2;
      margin: 0;
      font-weight: 500;
    }
  
    &__meta {
      display: flex;
      gap: 0.375rem;
      align-items: center;
      margin-top: 0.375rem;
      font-size: 0.75rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &__description {
      margin-top: 1rem;
      font-size: 0.75rem;
      line-height: 1rem;
      font-weight: normal;
      color: $gray-500;
    }
  
    &__date {
      display: flex;
      gap: 0.375rem;
      align-items: center;
      align-self: flex-start;
      padding: 0.375rem 0.625rem 0.375rem 0.75rem;
      margin-top: 1rem;
      font-size: 0.75rem;
      line-height: 1.25;
      background-color: $white;
      border-radius: 0.5rem;
      border: 1px solid $gray-200;
    }
  
    &__date-icon {
      object-fit: contain;
      width: 0.75rem;
      aspect-ratio: 1 / 1;
    }
  }
  </style>