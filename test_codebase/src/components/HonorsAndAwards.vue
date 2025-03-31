<template>
    <div class="scroll-reveal honors-awards">
      <div class="honors-awards__header">
        <div class="honors-awards__icon">
          <img
            loading="lazy"
            src="../assets/honors.svg"
            alt="Honors and Awards Icon"
            class="honors-awards__icon-image"
          />
        </div>
        <div class="honors-awards__title-container">
          <h2 class="honors-awards__title">Honors & Awards</h2>
          <p class="honors-awards__description">
            Acknowledgments for exceptional achievements and contributions in
            professional endeavors.
          </p>
        </div>
      </div>
      <div class="honors-awards__content">
        <AwardItem
          title="Role Modal Award"
          issuer="Modak Analytics"
          role="Senior Software Developer"
          description="As a key contributor to Project Nabu, I played a pivotal role in developing an innovative data engineering platform that significantly enhances data preparation efficiency. Our efforts were recognized when Modak Nabu™ was featured in Gartner's 'Market Guide for Data Preparation Tools' in 2020, highlighting its impact on accelerating data preparation by automating repetitive tasks with active metadata services."
          date="March 2021"
        />
        <AwardItem
          title="Participated in CTO talks"
          issuer="Keylane"
          role="Software Engineer"
          description="Participated in CTO discussions to present solutions for streamlining the deployment process by enabling front-end-editable functionality through a CMS."
          date="Jan 2023"
        />
      </div>
    </div>
  </template>
  
  <script>
  import AwardItem from './AwardItem.vue';
  
  export default {
    name: 'HonorsAndAwards',
    components: {
      AwardItem,
    },
  };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .honors-awards {
    display: flex;
    flex-direction: column;
    margin-top: 1rem;
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
    }
  
    &__icon-image {
      object-fit: contain;
      width: 1.5rem;
      aspect-ratio: 1 / 1;
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
      flex-wrap: wrap;
      gap: 1.5rem;
      padding: 2rem 1.5rem 2.5rem;
      width: 100%;
      font-weight: 500;
      color: $black;
    }
  
    @media (max-width: 768px) {
      &__header {
        padding: 1.25rem;
      }
  
      &__content {
        padding: 1.25rem;
      }
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
  </style>