<template>
    <a 
    href="https://www.linkedin.com/in/naveen-chilakala/"
    class="social-icon-wrapper"
    target="_blank">
      <img
        loading="lazy"
        :src="src"
        class="social-icon"
        alt="Social media icon"
      />
    </a>
  </template>
  
  <script>
  export default {
    name: 'SocialIcon',
    props: {
      src: {
        type: String,
        required: true
      }
    }
  }
  </script>
  
  <style lang="scss" scoped>
 @use '@/assets/main.scss' as *;
  .social-icon-wrapper {
    display: flex;
    gap: 0.625rem;
    justify-content: center;
    align-items: center;
    align-self: stretch;
    padding: 0.5rem;
    width: 2.5rem;
    cursor: pointer;
  }
  
  .social-icon {
    width: 1.5rem;
    aspect-ratio: 1;
    object-fit: contain;
  }
  </style>