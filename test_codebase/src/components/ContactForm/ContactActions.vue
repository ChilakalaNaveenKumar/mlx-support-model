<template>
    <div class="contact-actions">
      <div class="email-buttons">
      <button
        @click="copyEmail"
        class="email-button"
        role="button"
        tabindex="0"
        >
        <img
          loading="lazy"
          src="../../assets/mail.svg"
          class="email-icon"
          alt=""
        />
        <div class="self-stretch my-auto">
          <span class="text-white">Copy Email</span>
        </div>
      </button>

        <!-- Tooltip Notification -->
        <Teleport to="body">
          <div
            v-if="showTooltip"
            :style="{ top: tooltipY + 'px', left: tooltipX + 'px' }"
            class="tooltip-position px-2 py-1 bg-gray-800 text-white text-sm rounded shadow"
          >
            Copied to clipboard!
          </div>
        </Teleport>
      <button class="email-button" @click="sendEmail">
        <img
          loading="lazy"
          src="../../assets/mail.svg"
          class="email-icon"
          alt="Email icon"
        />
        <span>Send Email</span>
      </button>
    </div>
      <div class="social-icons">
        <SocialIcon v-for="(icon, index) in socialIcons" :key="index" :src="icon" />
      </div>
    </div>
  </template>
  
  <script>
  import SocialIcon from './SocialIcon.vue';
  import Linkden from '../../assets/linkden.svg';
  
  export default {
    name: 'ContactActions',
    components: {
      SocialIcon
    },
    data() {
      return {
        socialIcons: [
          Linkden
        ],
        showTooltip: false,
        tooltipX: 0,
        tooltipY: 0,
        email: "nchilaka1995@gmail.com"
      }
    },
    methods: {
      sendEmail() {
          const subject = "Hello!";
          const body = "I wanted to reach out regarding";

          // Construct mailto link
          const mailtoLink = `mailto:${this.email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

          // Open the user's default email client
          window.location.href = mailtoLink;
      },
      async copyEmail(event) {
        try {
          await navigator.clipboard.writeText(this.email);
  
          // Get cursor position
          this.tooltipX = event.pageX + 10;
          this.tooltipY = event.pageY + 60; // Offset above the cursor
  
          // Show tooltip
          this.showTooltip = true;
  
          // Hide tooltip after 1.5 seconds
          setTimeout(() => {
            this.showTooltip = false;
          }, 1500);
        } catch (err) {
          console.error("Failed to copy email:", err);
        }
      }
    },
  }
  </script>
  
  <style lang="scss" scoped>
  @use '@/assets/main.scss' as *;
  @use "sass:color";

  .tooltip-position {
  position: absolute;
  z-index: 10;
  pointer-events: none;
  color: black !important;
  background: white;
  border-radius: 5px;
  padding: 1rem;
  background: #dee8de;
  font-weight: 500;
}
  .email-buttons {
    display: flex;
    flex-direction: row;
    gap: 1rem;
  }
  .contact-actions {
    display: flex;
    flex-direction: column;
    align-items: center;
    align-self: center;
    span {
        color: $white;
    }
  }
  
  .email-button {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    align-items: center;
    padding: 1.5rem 3rem;
    font-size: 1.125rem;
    font-weight: 500;
    line-height: 1.375;
    color: $black;
    background-color: $neutral-800;
    border-radius: 0.75rem;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease;
  
    &:hover {
      background-color: color.scale($neutral-800, $lightness: -10%);
    }
  
    @media (max-width: 768px) {
      // padding-left: 1.25rem;
      // padding-right: 1.25rem;
      padding: 1rem 1rem;
      font-size: 0.8rem;
    }
  }
  
  .email-icon {
    width: 1.5rem;
    aspect-ratio: 1;
    object-fit: contain;
  }
  
  .social-icons {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    margin-top: 1.5rem;
  }
  </style>