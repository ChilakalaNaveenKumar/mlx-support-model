<template>
    <div class="work-experience-item">
      <div class="work-experience-item__marker"></div>
      <div class="work-experience-item__content">
        <div class="work-experience-item__header">
          <div class="work-experience-item__title-container">
            <h3 class="work-experience-item__title">{{ job.title }}</h3>
            <div class="work-experience-item__company-info">
              <span>{{ job.company }}</span>
              <span>{{ job.employmentType }}</span>
            </div>
          </div>
          <div class="work-experience-item__duration">
            <img
              loading="lazy"
              src="../assets/calendar.svg"
              alt="Calendar Icon"
              class="work-experience-item__duration-icon"
            />
            <span>{{ job.duration }}</span>
          </div>
        </div>
        <div class="work-experience-item__location">
          <img
            loading="lazy"
            src="../assets/location.svg"
            alt="Location Icon"
            class="work-experience-item__location-icon"
          />
          <span>{{ job.location }}</span>
        </div>
        <p :id="`experience-` + index" class="work-experience-item__description">
          {{ job.description }}
        </p>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'WorkExperienceItem',
    props: {
      index: {
        type: Number,
        required: true,
      },
      job: {
        type: Object,
        required: true,
      },
    },
    mounted() {
      const pTag = document.getElementById("experience-" + this.index);
      const text = pTag.innerHTML.trim();
      
      // Split based on double new lines or multiple spaces
      const points = text.split(/\n\s*\n/).map(point => point.trim());

      // Replace paragraph content with a list
      pTag.innerHTML = `
        <p>${points[0]}</p> 
        <ul>${points.slice(1).map(point => `<li class="bullet-points">${point}</li>`).join("")}</ul>
      `
    },
  };
  </script>
  
  <style lang="scss">
   @use '../assets/main.scss' as *;
  .work-experience-item {
    display: flex;
    gap: 1.5rem;
    &:last-child {
      margin-bottom: 0;
    }
  
    &__marker {
        flex-shrink: 0;
        width: 0.625rem;
        height: 0.625rem;
        border: 2px solid #FFFFFF;
        background-color: #77777D;
        border-radius: 50%;
        margin-top: 1.25rem;
        margin-left: 0.5rem;
        z-index: 20;
    }
  
    &__content {
      flex: 1;
      min-width: 15rem;
    }
  
    &__header {
      display: flex;
      flex-wrap: wrap;
      gap: 1.5rem;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 1rem;
    }
  
    &__title-container {
      flex: 1;
      min-width: 15rem;
    }
  
    &__title {
      font-size: 1rem;
      font-weight: 500;
      line-height: 1.25;
      color: $black;
    }
  
    &__company-info {
      display: flex;
      gap: 0.375rem;
      margin-top: 0.375rem;
      font-size: 0.75rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &__duration {
      display: flex;
      gap: 0.375rem;
      align-items: center;
      padding: 0.375rem 0.625rem;
      background-color: $white;
      border: 1px solid $black;
      border-radius: 0.5rem;
      font-size: 0.75rem;
      line-height: 1.25;
    }
  
    &__duration-icon {
      width: 0.75rem;
      height: 0.75rem;
      object-fit: contain;
    }
  
    &__location {
      display: flex;
      gap: 0.375rem;
      align-items: center;
      margin-top: 1rem;
      font-size: 0.75rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &__location-icon {
      width: 0.875rem;
      height: 0.875rem;
      object-fit: contain;
    }
  
    &__description {
      margin-top: 1rem;
      font-size: 0.75rem;
      line-height: 1.25;
      color: $gray-500;
      max-width: 25rem;
    }
  }

  .bullet-points {
    margin-bottom: 0.5rem;
  }
  
  @media (max-width: 768px) {
    .work-experience-item {
      gap: 1rem;
  
      &__header {
        flex-direction: column;
        gap: 1rem;
      }
  
      &__duration {
        align-self: flex-start;
      }
    }
  }

  @media (max-width: 768px) {
    .work-experience-item {
        &__marker {
            margin-left: -0.2rem;
        }
    }
  }
  </style>