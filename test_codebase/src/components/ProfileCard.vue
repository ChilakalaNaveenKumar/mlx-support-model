<template>
    <div class="scroll-reveal profile-card">
      <img
        loading="lazy"
        src="../assets/profile.jpeg"
        class="profile-image"
        alt="Profile image"
      />
      <div class="profile-details">
        <div class="profile-icon-container">
          <img
            loading="lazy"
            src="../assets/person.svg"
            class="profile-icon"
            alt="Profile icon"
          />
        </div>
        <div class="profile-content">
          <div class="profile-header">
            <h2 class="profile-title">Professional Profile</h2>
            <p class="profile-description">
              Innovative Senior Software Developer with 7+ years of experience in
              designing, developing, and optimizing scalable web applications. My
              expertise spans front-end frameworks (Vue.js, Nuxt.js, React,
              Angular) and backend technologies (Java, Spring Boot, Python,
              Django, Flask), allowing me to build full-stack solutions with high
              performance and modularity.<br /><br/>I thrive in fast-paced,
              problem-solving environments, having independently managed global
              clients like Keylane (Netherlands) while working remotely from
              India. My ability to analyze, troubleshoot, and deploy solutions
              autonomously has helped me lead projects across enterprise-level
              companies and startups.
            </p>
          </div>
          <div class="profile-location">
            <img
              loading="lazy"
              src="../assets/location.svg"
              class="location-icon"
              alt="Location icon"
            />
            <span class="location-text">Montgomery, Alabama, USA.</span>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  export default {
    name: 'ProfileCard'
  }
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  .profile-card {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    max-width: 720px;
    margin: 50px auto 0;
    @media screen and (max-width: 719px) {
      flex-direction: column;
      align-items: center;
      width: 80%;
    }
  }
  
  .profile-image {
    aspect-ratio: 0.53;
    min-width: 240px;
    max-width: 263px;
  }
  
  .profile-details {
    display: flex;
    flex-direction: column;
    flex: 1;
    align-self: flex-start;
    padding: 1.5rem;
    background-color: $white;
    border-radius: 1rem;
    border: 1px solid $gray-200;
    min-width: 240px;
    height: 100%;
  }
  
  .profile-icon-container {
    display: flex;
    gap: 0.625rem;
    align-items: center;
    padding: 0.875rem;
    border-radius: 0.75rem;
    border: 1px solid $gray-200;
    background-color: $gray-100;
    height: 52px;
    width: 52px;
  }
  
  .profile-icon {
    object-fit: contain;
    width: 1.5rem;
    aspect-ratio: 1 / 1;
  }
  
  .profile-content {
    display: flex;
    flex-direction: column;
    margin-top: 3rem;
    width: 100%;
    font-weight: 500;
    color: black;
  }
  
  .profile-header {
    display: flex;
    flex-direction: column;
    width: 100%;
  }
  
  .profile-title {
    font-size: 1.25rem;
    line-height: 1.25;
    font-weight: 500;
  }
  
  .profile-description {
    margin-top: 0.75rem;
    font-size: 0.875rem;
    line-height: 1.25;
    color: $gray-500;
  }
  
  .profile-location {
    display: flex;
    gap: 0.375rem;
    align-items: center;
    align-self: flex-start;
    margin-top: 1.5rem;
    font-size: 0.75rem;
    line-height: 1.25;
  }
  
  .location-icon {
    object-fit: contain;
    flex-shrink: 0;
    align-self: stretch;
    margin: auto 0;
    width: 0.875rem;
    aspect-ratio: 1 / 1;
  }
  
  .location-text {
    align-self: stretch;
    margin: auto 0;
  }
  
  @media (max-width: 768px) {
    .profile-details {
      padding: 1.25rem;
      max-width: 100%;
    }
  
    .profile-content {
      margin-top: 2.5rem;
    }
  }
  </style>