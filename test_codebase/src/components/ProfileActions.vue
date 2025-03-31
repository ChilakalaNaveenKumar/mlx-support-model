<template>
    <div
      class="scroll-reveal flex flex-wrap justify-center items-start w-full text-sm tracking-tight max-md:max-w-full"
    >
      <a
        href="#"
        @click.prevent="copyEmail"
        class="flex justify-center items-center px-4 py-2.5 text-black rounded-xl bg-neutral-800 mr-2 mb-2"
        role="button"
        tabindex="0"
        >
        <img
          loading="lazy"
          src="../assets/mail.svg"
          class="object-contain shrink-0 self-stretch my-auto w-4 aspect-square mr-2"
          alt=""
        />
        <div class="self-stretch my-auto">
          <span class="text-white">Copy Email</span>
        </div>
        </a>

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
        <a
        href="/resume-template/Senior-Fullstack-developer-VueJs.pdf"
        download="Naveen_Kumar_Chilakala_CV.pdf"
        class="flex justify-center items-center px-4 py-2.5 text-black rounded-xl bg-neutral-200 mr-2 mb-2"
        role="button"
        tabindex="0"
      >
        <img
          loading="lazy"
          src="../assets/download.svg"
          class="object-contain shrink-0 self-stretch my-auto w-4 aspect-square mr-2"
          alt=""
        />
        <div class="self-stretch my-auto">Download CV</div>
      </a>
    </div>
  </template>
  
  <script setup>
    import { ref } from "vue";

const email = ref("nchilaka1995@gmail.com");
const showTooltip = ref(false);
const tooltipX = ref(0);
const tooltipY = ref(0);

const copyEmail = async (event) => {
  try {
    await navigator.clipboard.writeText(email.value);

    // Get cursor position
    tooltipX.value = event.clientX + 10; 
    tooltipY.value = event.clientY + 30; // Offset above the cursor

    // Show tooltip
    showTooltip.value = true;

    // Hide tooltip after 1.5 seconds
    setTimeout(() => {
      showTooltip.value = false;
        }, 1500);
      } catch (err) {
        console.error("Failed to copy email:", err);
      }
    };
  </script>
  
  <style lang="scss" scoped>
  @use '../assets/main.scss' as *;
  /* Ensure tooltip is positioned properly */
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
  .flex {
    display: flex;
  }
  
  .flex-wrap {
    flex-wrap: wrap;
  }
  
  .justify-center {
    justify-content: center;
  }
  
  .items-start {
    align-items: flex-start;
  }
  
  .w-full {
    width: 100%;
  }
  
  .text-sm {
    font-size: 0.875rem;
    line-height: 1.25rem;
  }
  
  .max-md\:max-w-full {
    @media (max-width: 767px) {
      max-width: 100%;
    }
  }
  
  .px-4 {
    padding-left: 1rem;
    padding-right: 1rem;
  }
  
  .py-2\.5 {
    padding-top: 0.625rem;
    padding-bottom: 0.625rem;
  }
  
  .text-black {
    color: #000;
  }
  
  .rounded-xl {
    border-radius: 0.75rem;
  }
  
  .bg-neutral-800 {
    background-color: $neutral-800;
  }
  
  .bg-neutral-200 {
    background-color: $gray-700;
  }
  
  .mr-2 {
    margin-right: 0.5rem;
  }
  
  .mb-2 {
    margin-bottom: 0.5rem;
  }
  
  .object-contain {
    object-fit: contain;
  }
  
  .shrink-0 {
    flex-shrink: 0;
  }
  
  .self-stretch {
    align-self: stretch;
  }
  
  .my-auto {
    margin-top: auto;
    margin-bottom: auto;
  }
  
  .w-4 {
    width: 1rem;
  }
  
  .aspect-square {
    aspect-ratio: 1 / 1;
  }
  
  .text-white {
    color: $white;
  }
  </style>