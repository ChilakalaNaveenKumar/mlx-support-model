<template>
    <div class="scroll-reveal worked-with">
      <div class="worked-with__header">
        Worked With
      </div>
      <div class="worked-with__logos" ref="logoContainer">
        <div class="worked-with__logos-inner" :style="{ transform: `translateX(${scrollPosition}px)` }">
          <img
            v-for="(logo, index) in logos"
            :key="index"
            :src="logo"
            :alt="`Company logo ${index + 1}`"
            class="worked-with__logo"
          />
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import modakanalytics from '@/assets/modakanalytics.svg';
  import keylane from '@/assets/keylane.svg';
  import lifevantage from '@/assets/lifevantage.svg';
  import xebia from '@/assets/xebia.svg';
  export default {
    name: 'WorkedWith',
    data() {
      return {
        logos: [
          modakanalytics,
          keylane,
          lifevantage,
          xebia
        ],
        scrollPosition: 0,
        scrollSpeed: 0.8,
        animationFrameId: null,
        scrollDirection: 1
      }
    },
    mounted() {
      this.startScrollAnimation()
    },
    beforeUnmount() {
      this.stopScrollAnimation()
    },
    methods: {
      startScrollAnimation() {
        const animate = () => {
          this.scrollPosition -= this.scrollSpeed * this.scrollDirection
          const containerWidth = this.$refs.logoContainer.offsetWidth
          const innerWidth = this.$refs.logoContainer.scrollWidth
          if (Math.abs(this.scrollPosition) >= innerWidth - containerWidth || this.scrollPosition > 0) {
            this.scrollDirection *= -1
          }
          this.animationFrameId = requestAnimationFrame(animate)
        }
        this.animationFrameId = requestAnimationFrame(animate)
      },
      stopScrollAnimation() {
        if (this.animationFrameId) {
          cancelAnimationFrame(this.animationFrameId)
        }
      }
    }
  }
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .worked-with {
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: 1rem;
    width: 100%;
    background-color: $white;
    border-radius: 1rem;
    border: 1px solid $gray-200;
  
    &__header {
      flex: 1;
      padding: 1.25rem 1.5rem;
      width: 100%;
      font-size: 0.875rem;
      font-weight: 500;
      line-height: 1.25;
      text-align: center;
      color: $black;
      border-bottom: 1px solid $gray-200;
    }
  
    &__logos {
      padding: 2rem 1.5rem;
      width: 100%;
      overflow: hidden;
    }
  
    &__logos-inner {
      display: flex;
      gap: 2.5rem;
      transition: transform 0.1s linear;
    }
  
    &__logo {
      object-fit: contain;
      flex-shrink: 0;
      align-self: stretch;
      margin: auto 0;
      width: 108px;
      aspect-ratio: 4.69;
    }
  }
  
  @media (max-width: 720px) {
    .worked-with {
      max-width: 80%;
      margin: 0px auto;
      margin-top: 24px;
      &__logos-inner {
        justify-content: flex-start;
      }
    }
  }
  
  @media (min-width: 720px) {
    .worked-with {
      max-width: 720px;
      margin: 0px auto;
      margin-top: 24px;
      &__logos-inner {
        justify-content: flex-start;
      }
    }
  }
  </style>