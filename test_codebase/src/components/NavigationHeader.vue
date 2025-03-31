<template>
  <nav role="navigation" aria-label="Main navigation" class="nav-header">
    <img loading="lazy" src="../assets/closeup.jpeg" alt="" class="logo" />
    <button
      class="menu-toggle"
      @click="toggleMenu"
      :aria-expanded="isMenuOpen.toString()"
      aria-controls="mobile-menu"
    >
      <span class="sr-only">Toggle menu</span>
      <img
        v-if="!isMenuOpen"
        loading="lazy"
        src="../assets/hamburger-menu.svg"
        alt=""
        class="menu-icon"
      />
      <img
        v-if="isMenuOpen"
        loading="lazy"
        src="../assets/hamburger-close.svg"
        class="object-contain w-5 aspect-square"
        alt=""
      />
    </button>
    <div id="mobile-menu" class="mobile-menu" :class="{ 'active': isMenuOpen }">
      <div class="nav-items">
        <a
          v-for="tab in tabs"
          :key="tab.id"
          :href="`#${tab.id}`"
          role="menuitem"
          class="nav-item"
          :class="{ 'active': isActiveTab(tab.id) }"
          @click.prevent="setActiveTab(tab.id)"
        >
          {{ tab.name }}
        </a>
      </div>
      <button
        :class="{ 'active': count > 0 }"
        class="contact-button"
        @click="increment"
      >
        <img loading="lazy" src="../assets/CalligraphyPen.svg" alt="" class="contact-icon" />
        <span>Contact Me</span>
      </button>
    </div>
  </nav>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue';

const tabs = [
  { id: 'home', name: 'Home' },
  { id: 'profile', name: 'Profile' },
  { id: 'experience', name: 'Experience' },
  { id: 'education', name: 'Education' },
  { id: 'skills', name: 'Skills' }
];
const activeTab = ref('home');
const count = ref(0);
const isMenuOpen = ref(false);

const setActiveTab = (tabId) => {
  activeTab.value = tabId;
  isMenuOpen.value = false;

  // Find the target section
  const section = document.getElementById(tabId);

  if (section) {
    // Adjust for fixed navbar height (e.g., 70px)
    const offset = 70;
    const sectionPosition = section.getBoundingClientRect().top + window.scrollY - offset;

    // Smooth scroll
    window.scrollTo({ top: tabId == "home" ? 0 : sectionPosition, behavior: "smooth" });
  }
};

const isActiveTab = (tabId) => {
  return activeTab.value === tabId;
};

const increment = () => {
  let tabId = 'contactme';
  const section = document.getElementById(tabId);
  const offset = 70;
  const sectionPosition = section.getBoundingClientRect().top + window.scrollY - offset;

  // Smooth scroll
  window.scrollTo({ top: sectionPosition, behavior: "smooth" });
};

const toggleMenu = () => {
  isMenuOpen.value = !isMenuOpen.value;
};

const handleScroll = () => {
  const sections = tabs.map(tab => document.getElementById(tab.id));
  const scrollPosition = window.scrollY + window.innerHeight / 3;

  for (let i = sections.length - 1; i >= 0; i--) {
    const section = sections[i];

    // Check if the section is visible in the viewport
    if (section) {
      const sectionTop = section.offsetTop - 80; // Adjust for fixed navbar height

      if (scrollPosition >= sectionTop) {
        if (activeTab.value !== tabs[i].id) {
          activeTab.value = tabs[i].id;
        }
        break;
      }
    }
  }
};


onMounted(() => {
  window.addEventListener('scroll', handleScroll);
  handleScroll();
});

onUnmounted(() => {
  window.removeEventListener('scroll', handleScroll);
});
</script>

<style lang="scss" scoped>
@use '../assets/main.scss' as *;

.nav-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.375rem;
  width: 80%;
  max-width: 720px;
  font-size: 0.875rem;
  font-weight: 500;
  line-height: 1.25;
  background-color: $white;
  border-bottom: 1px solid $gray-200;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  position: fixed;
  border-radius: 20px;
  top: 50px;
  
  transform: translateX(-50%);
  z-index: 1000;
  left: 50%;
  transform: translateX(-50%);
  @media (max-width: 720px) {
    // left: 10vw;
    width: 80%;
  }
  @media (min-width: 720px) {
    width: 720px;
    left: 50%;
    transform: translateX(-50%);
  }
}

.logo {
  object-fit: contain;
  width: 2.5rem;
  aspect-ratio: 1 / 1;
  border-radius: 50%;
}

.menu-toggle {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 2.5rem;
  height: 2.5rem;
  background-color: $black;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
}

.menu-icon {
  width: 1.5rem;
  height: 1.5rem;
}

.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border-width: 0;
}

.mobile-menu {
  display: none;
  position: absolute;
  left: 0;
  right: 0;
  top: 110%;
  background-color: $white;
  padding: 1rem;
  border-radius: 0 0 1rem 1rem;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
  z-index: 10;

  @media screen and (max-width: 719px) {
    &.active {
      display: block;
    }
  }
}

.nav-items {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.nav-item {
  padding: 0.625rem 0.75rem;
  background-color: $white;
  border-radius: 0.75rem;
  cursor: pointer;
  text-decoration: none;
  color: $black;

  &.active {
    background-color: $gray-100;
  }
}

.contact-button {
  display: flex;
  gap: 0.5rem;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding: 0.625rem 1rem;
  color: $white;
  border-radius: 0.75rem;
  background-color: $neutral-800;
  border: none;
  cursor: pointer;

  &.active {
    opacity: 0.8;
  }
}

.contact-icon {
  width: 1rem;
  aspect-ratio: 1 / 1;
}

@media (min-width: 720px) {
  .nav-header {
    flex-wrap: wrap;
    gap: 2.5rem;
    max-width: 720px;
    margin: 0 auto;
    padding: 0.375rem 1rem;
  }

  .menu-toggle {
    display: none;
  }

  .mobile-menu {
    display: flex;
    position: static;
    padding: 0;
    box-shadow: none;
    background-color: transparent;
  }

  .nav-items {
    flex-direction: row;
    margin-bottom: 0;
    min-width: 240px;
  }

  .contact-button {
    width: auto;
  }
}
</style>