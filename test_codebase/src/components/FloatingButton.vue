<template>
    <div
      class="floating-button"
      :class="{ 'floating-button--visible': isVisible }"
       @click="sendEmail"
    >
      <img
        loading="lazy"
        src="../assets/mail.svg"
        class="floating-button__icon"
        alt="Floating button icon"
      />
    </div>
  </template>
  
  <script>
  import { ref, onMounted, onUnmounted } from 'vue';
  
  export default {
    name: 'FloatingButton',
    setup() {
      const isVisible = ref(false);
      let lastScrollTop = 0;
  
      // const handleScroll = () => {
      //   const st = window.pageYOffset || document.documentElement.scrollTop;
      //   if (st > lastScrollTop && st > 100) {
      //     isVisible.value = true;
      //   } else if (st < lastScrollTop) {
      //     isVisible.value = false;
      //   }
      //   lastScrollTop = st <= 0 ? 0 : st;
      // };

      const handleScroll = () => {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        isVisible.value = scrollTop >= 50; // Show button only if scrollTop is 50 or more
      };
  
      onMounted(() => {
        window.addEventListener('scroll', handleScroll);
      });
  
      onUnmounted(() => {
        window.removeEventListener('scroll', handleScroll);
      });
      const sendEmail = () => {
        const email = "nchilaka1995@gmail.com"; // Replace with your email
        const subject = "Hello!";
        const body = "I wanted to reach out regarding...";

        // Construct mailto link
        const mailtoLink = `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

        // Open the user's default email client
        window.location.href = mailtoLink;
      };
  
      return {
        isVisible,
        sendEmail
      };
    }
  };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .floating-button {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 3.5rem;
    height: 3.5rem;
    background-color: $neutral-800;
    border-radius: 50%;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    opacity: 0;
    transform: translateY(100%);
    transition: opacity 0.3s ease, transform 0.3s ease;
    cursor: pointer;
  
    &--visible {
      opacity: 1;
      transform: translateY(0);
    }
  
    &__icon {
      width: 1.5rem;
      height: 1.5rem;
      object-fit: contain;
    }
  }
  
  
    .floating-button {
      bottom: 50%;
      right: 1rem;
    }
  
  </style>