<template>
    <div class="education-item">
      <div class="education-details">
        <div class="school-name">{{ school }}</div>
        <div class="degree-info">
          <span>{{ degree }}</span>
          <span>{{ field }}</span>
        </div>
      </div>
      <div class="education-date">
        <img loading="lazy" src="../assets/calendar.svg" alt="Calendar icon" />
        <span>{{ startDate }} - {{ endDate }}</span>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'EducationItem',
    props: {
      school: {
        type: String,
        required: true,
      },
      degree: {
        type: String,
        required: true,
      },
      field: {
        type: String,
        required: true,
      },
      startDate: {
        type: String,
        required: true,
      },
      endDate: {
        type: String,
        required: true,
      },
    },
  };
  </script>
  
  <style lang="scss" scoped>
   @use '../assets/main.scss' as *;
  .education-item {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    align-items: flex-start;
    width: 100%;
    margin-top: 10px;
    &:first-child {
      margin-top: 0;
    }
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  
  .education-details {
    display: flex;
    flex-direction: column;
    flex: 1;
    min-height: 42px;
    min-width: 240px;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  
  .school-name {
    font-size: 1rem;
    line-height: 1.25;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  
  .degree-info {
    display: flex;
    gap: 0.375rem;
    align-items: center;
    align-self: flex-start;
    margin-top: 0.375rem;
    font-size: 0.75rem;
    line-height: 1.25;
    color: $gray-500;
  
    span {
      align-self: stretch;
      margin: auto 0;
    }
  }
  
  .education-date {
    display: flex;
    gap: 0.375rem;
    align-items: center;
    padding: 0.375rem 0.625rem 0.375rem 0.75rem;
    font-size: 0.75rem;
    line-height: 1.25;
    background-color: $white;
    border-radius: 0.5rem;
    border: 1px solid $gray-200;
  
    img {
      object-fit: contain;
      flex-shrink: 0;
      align-self: stretch;
      margin: auto 0;
      width: 0.75rem;
      aspect-ratio: 1;
    }
  
    span {
      align-self: stretch;
      margin: auto 0;
    }
  }
  </style>