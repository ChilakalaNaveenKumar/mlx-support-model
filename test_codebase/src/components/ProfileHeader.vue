<template>
  <div class="profile-header">
    <div
      class="status-indicator"
      role="status"
      aria-live="polite"
    >
      <div
        class="status-dot"
        aria-hidden="true"
      >
        <div class="small-status-dot"></div>
      </div>
      <div class="status-text">Available for Work</div>
    </div>
    <h1 class="name">
      Naveen Kumar Chilakala
    </h1>
    <p class="title">
      Senior Fullstack Developer I Team Lead
    </p>
  </div>
</template>

<script>
export default {
  name: 'ProfileHeader'
}
</script>

<style lang="scss" scoped>
@use '../assets/main.scss' as *;

.profile-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 80%;
  max-width: 720px;
  margin: 0 auto;
  color: $black;
  padding: 1rem;
  margin-top: 150px;

  @media (min-width: 768px) {
    padding: 2rem 0;
  }

  .status-indicator {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    padding: 6px 10px 6px 12px;
    font-size: 0.75rem;
    line-height: 1rem;
    background-color: $white;
    border-radius: 0.5rem;
    border: 1px solid $gray-200;

    .status-dot {
      display: flex;
      gap: 0.625rem;
      align-self: stretch;
      padding: 2px 0;
      margin: auto 0;
      width: 0.5rem;
      height: 0.5rem;
      min-height: 8px;
      background-color: #BEECDC;
      border-radius: 100px;
      animation: blink 1s infinite;
      .small-status-dot {
        width: 0.25rem;
        height: 0.25rem;
        background-color: #10B981;
        margin: 0rem auto;
        border-radius: 50%;
      }
    }

    .status-text {
      align-self: stretch;
      margin: auto 0;
    }
  }

  .name {
    align-self: stretch;
    margin-top: 0.5rem;
    font-size: 2rem;
    font-weight: 500;
    line-height: 1.2;
    text-align: center;
    margin-bottom: 0.5rem;

    @media (min-width: 768px) {
      font-size: 3rem;
      line-height: 1;
    }
  }

  .title {
    margin-top: 0.5rem;
    font-size: 1rem;
    line-height: 1.5;
    text-align: center;
    max-width: 100%;
    color: $gray-500;
    font-weight: 500;

    @media (min-width: 768px) {
      font-size: 1.125rem;
      line-height: 1.75rem;
      max-width: 424px;
    }
  }
}

@keyframes blink {
  0% { opacity: 0; }
  50% { opacity: 1; }
  100% { opacity: 0; }
}
</style>