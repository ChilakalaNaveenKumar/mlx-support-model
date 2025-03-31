<template>
    <div class="contact-header">
      <img
        loading="lazy"
        src="../../assets/closeup.jpeg"
        class="contact-icon"
        alt="Contact icon"
      />
      <div class="contact-text">
        <h2 class="contact-title">Feel Free to Reach Out</h2>
        <p class="contact-subtitle">Let's start a productive conversation</p>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'ContactHeader'
  }
  </script>
  
  <style lang="scss" scoped>
  @use '@/assets/main.scss' as *;
  .contact-header {
    display: flex;
    flex-direction: column;
    width: 100%;
  }
  
  .contact-icon {
    align-self: center;
    width: 5rem;
    aspect-ratio: 1;
    object-fit: contain;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border-radius: 50%;
  }
  
  .contact-text {
    display: flex;
    flex-direction: column;
    margin-top: 1.5rem;
    width: 100%;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  
  .contact-title {
    font-size: 1.25rem;
    line-height: 1.25;
    letter-spacing: -0.025em;
    font-weight: 500;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  
  .contact-subtitle {
    align-self: center;
    margin-top: 0.25rem;
    font-size: 1.125rem;
    line-height: 1.375;
    max-width: 424px;
    color: $gray-500;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  </style>