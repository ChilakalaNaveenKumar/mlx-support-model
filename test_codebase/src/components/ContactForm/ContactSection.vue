<template>
    <section class="scroll-reveal contact-section">
      <div class="contact-container">
        <ContactHeader />
        <ContactActions />
      </div>
    </section>
  </template>
  
  <script>
  import ContactHeader from './ContactHeader.vue';
  import ContactActions from './ContactActions.vue';
  
  export default {
    name: 'ContactSection',
    components: {
      ContactHeader,
      ContactActions
    }
  }
  </script>
  
  <style lang="scss" scoped>
 @use '@/assets/main.scss' as *;
  .contact-section {
    display: flex;
    flex-direction: column;
    margin: 0px auto;
    margin-top: 50px;
    max-width: 720px;
  
    @media (max-width: 768px) {
      max-width: 80%;
    }
  }
  
  .contact-container {
    display: flex;
    flex-direction: column;
    width: 100%;
    font-weight: 500;
    text-align: center;
    color: $black;
  
    @media (max-width: 768px) {
      max-width: 100%;
    }
  }
  </style>