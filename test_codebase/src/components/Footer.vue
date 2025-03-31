<template>
    <footer class="scroll-reveal footer">
      <div class="footer__content">
        <div class="footer__copyright">© 2025, All Right Reserved</div>
        <div class="footer__licenses">Licenses</div>
      </div>
    </footer>
  </template>
  
  <style lang="scss" scoped>
  @use '@/assets/main.scss' as *;
  .footer {
    display: flex;
    flex-direction: column;
    padding-top: 1.5rem;
    margin: 0px auto;
    margin-top: 24px;
    max-width: 100%;
    font-size: 0.875rem;
    font-weight: 500;
    line-height: 1.25;
    text-align: center;
    color: $black;
    border-top: 1px solid $gray-200;

  
    @media (max-width: 768px) {
      max-width: 80%;
    }

    @media (min-width: 768px) {
      max-width: 720px;
        
    }
  
    &__content {
      width: 100%;
    }
  
    &__copyright {
      @media (max-width: 768px) {
        max-width: 100%;
      }
    }
  
    &__licenses {
      margin-top: 0.75rem;
  
      @media (max-width: 768px) {
        max-width: 100%;
      }
    }
  }
  </style>