<template>
    <div class="scroll-reveal work-experience">
      <div class="work-experience__header">
        <div class="work-experience__icon">
          <img
            loading="lazy"
            src="../assets/sparkle.svg"
            alt="Work Experience Icon"
            class="work-experience__icon-image"
          />
        </div>
        <div class="work-experience__title-container">
          <h2 class="work-experience__title">Work Experience</h2>
          <p class="work-experience__description">
            Proven track record in full-stack development, delivering impactful
            insights and results across industries.
          </p>
        </div>
      </div>
      <div class="work-experience__timeline">
        <div class="work-experience__timeline-line"></div>
        <WorkExperienceItem
          v-for="(job, index) in jobs"
          :key="index"
          :job="job"
          :index="index"
        />
      </div>
    </div>
  </template>
  
  <script>
  import WorkExperienceItem from './WorkExperienceItem.vue';
  
  export default {
    name: 'WorkExperience',
    components: {
      WorkExperienceItem,
    },
    data() {
      return {
        jobs: [
          {
            title: 'Senior Software Developer',
            company: 'LifeVantage (Computer Power Group)',
            employmentType: 'Full-Time',
            duration: 'April 2023 - Present',
            location: 'Montgomery, Alabama, USA',
            description: `Enhancing front-end development and payment integrations with Wordpress, Vue.js and state management, ensuring seamless user experiences. Bridging technical development and business needs with secure, high-performance solutions. Driving innovation through CI/CD automation and performance optimization for scalable deployments.
            
                Designed modular components using Vue to ensure reusability and separation of
                concerns across different modules.

                Applied Vue.js directives to implement a multi-level navigation system where parent
                routes dynamically rendered child components based on the user context.

                Integrated a payment gateway with Apple Pay and Google Pay, including complete
                order information and card details modification for seamless transactions using
                Spring Boot Frame Work.

                Implemented an REST API auto-populate feature for addresses based on the entered
                pin code, minimizing user input and enhancing interaction efficiency.

                Developed hover effects, fade-in/out transitions, and page animations with hardware-
                accelerated CSS properties (transform and opacity) and requestAnimationFrame.

                Implemented a dynamic box modal to structure content while maintaining a clean
                UI layout.

                Integrated Stripe into a content management platform to process recurring
                subscription payments securely while ensuring PCI-DSS compliance. This streamlined
                the user experience by allowing quick checkouts and providing real-time updates on
                payment statuses.

                Proficient in managing SQLite3 server for lightweight database solutions, ensuring
                high performance and data integrity.
            `,
          },
          {
            title: 'Software Engineer',
            company: 'Keylane. Netherlands (CoMakeIT)',
            employmentType: 'Fulltime',
            duration: 'May 2021 - March 2023',
            location: 'Hyderabad, India',
            description: `Developed scalable front-end solutions using Nuxt.js and Vue.js, optimizing performance, routing, and state management for enterprise applications. Led UI/UX improvements, caching strategies, and code optimizations to enhance application speed and user experience. Collaborated with cross-functional teams and CTO discussions, implementing headless CMS integrations and secure payment processing.
                    
                    Architected and implemented a CMS functionality with Vue.js, Vuex, and Springboot
                    for content management, integrating it with SQLite3 to fetch and manage content via
                    REST APIs.
                    
                    Managed the coordination of services using a Docker container platform with
                    Kubernetes.
                    
                    Participated in discussions with the CTO, presenting innovative solutions to
                    streamline content editing processes, such as updating images, text, and links directly
                    within the application, replacing traditional, time-consuming methods.
                    
                    Provided centralized solution to handle complex choices in pension system using
                    Redux as the central data store module.
                    
                    Configured Kubernetes-managed load balancer to automatically distribute
                    incoming traffic across multiple containers in the AKS cluster, ensuring availability and
                    scaling.
                   
                    Optimized JavaScript bundles by implementing code splitting and lazy loading,
                    ensuring only essential modules are loaded during initial rendering.
                    
                    Implemented a Headless CMS with Spring Boot for content management, integrating
                    it with GraphQL to fetch and manage content via REST APIs.
                    
                    Configured HPA for applications to dynamically adjust the number of pods. This
                    ensured optimal resource usage during traffic spikes and maintained performance
                    stability during periods of high demand.
                    
                    Integrated Stripe with Spring Boot and Hibernate to facilitate secure payment
                    processing, efficiently managing customer transactions and ensuring compliance with
                    payment standards.
                    
                    Managed business objects and resource management in the business logic tier using
                    Spring Framework.
                    
                    Configure the CI/CD pipeline in Azure DevOps to automatically run the test suite after
                    every code change.
                    
                    Designed and implemented dynamic, visually appealing web interfaces using Tailwind
                    CSS, streamlining design processes and ensuring responsive layouts.
                    
                    Implemented semantic HTML for screen readers, providing keyboard navigation,
                    maintaining color contrast ratios,and testing accessibility tools to comply with Level
                    AA standards.
            
            `,
          },
          {
            title: 'Senior Software Developer I Team Lead',
            company: 'Clients GSK, Humana. USA (Modak Analytics LLP)',
            employmentType: 'Full-Time',
            duration: 'Oct 2017 - May 2021',
            location: 'Hyderabad, India',
            description: `Engineered SEO-friendly, server-side rendered (SSR) applications using Nuxt.js and Vue.js, optimizing performance and scalability. Developed dynamic UI components, automated testing workflows, and CI/CD pipelines to streamline deployment. Led Kubernetes-based containerized deployments, enhancing system reliability and scalability for enterprise data solutions.
            
            Developed a reusable Vue.js component and packaged it as a private NPM plugin,
            enabling global component injection and improving modularity.
            
            Distributed the plugin via a private repository and local NPM registry, allowing
            seamless integration across multiple Vue projects.
            
            Architected a framework for a Vue.js editor that eliminates the need for a traditional
            CI/CD deployment cycle. Changes are deployed dynamically, streamlining the
            development process.
            
            Crafted an intelligent web SQL editor with real-time, context-aware suggestions
            based on user input using the CodeMirror plugin. Enhanced user experience and
            productivity with dynamic, accurate recommendations.
            
            Developed dynamic forms and tables rendering various components using HTML
            and JavaScript as string objects. Improved flexibility and user experience with dynamic
            content generation.
            
            Implemented CI/CD pipelines using Azure DevOps for seamless integration and
            deployment. Streamlined deployment processes with Azure agents, enhancing
            efficiency and reliability.
            
            Designed and implemented a real-time visualization tool for client executives to
            monitor job performance and status. Enhanced operational efficiency with interactive
            charts, automated alerts, and custom reports.
            
            Implemented semantic HTML for screen readers, providing keyboard navigation,
            maintaining color contrast ratios, and testing accessibility tools to comply with Level
            AA standards.
            
            Automated end-to-end testing with Node.js and Puppeteer, allowing testers to
            execute test cases using JSON input without writing code.
            
            Integrated a fully automated testing workflow, reducing manual effort by dynamically
            generating test scenarios and running them in a CI/CD pipeline.
            
            Leveraged YAML-based configurations to define custom triggers, parallel job
            execution, and environment-specific deployment strategies, ensuring streamlined
            releases and reduced downtime during updates.
            
            Designed and developed Microservices Architecture to divide application into
            business components using Spring Boot.
            
            Automated testing workflows using Puppeteer and Node.js to run parallel browser
            instances, enabling efficient end-to-end testing across Chrome environments.`,
          },
        ],
      };
    },
  };
  </script>
  
  <style lang="scss" scoped>
   @use '../assets/main.scss' as *;
  .work-experience {
    margin-top: 1rem;
    background-color: $white;
    border-radius: 1rem;
    border: 1px solid $gray-200;
    overflow: hidden;
  
    &__header {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      padding: 1.5rem;
      border-bottom: 1px solid $gray-200;
    }
  
    &__icon {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 3.25rem;
      height: 3.25rem;
      background-color: $gray-100;
      border: 1px solid $gray-200;
      border-radius: 0.75rem;
    }
  
    &__icon-image {
      width: 1.5rem;
      height: 1.5rem;
      object-fit: contain;
    }
  
    &__title-container {
      margin-top: 1.5rem;
      max-width: 25rem;
    }
  
    &__title {
      font-size: 1.25rem;
      font-weight: 500;
      line-height: 1.2;
      color: $black;
    }
  
    &__description {
      margin-top: 0.75rem;
      font-size: 0.875rem;
      line-height: 1.25;
      color: $gray-500;
    }
  
    &__timeline {
      position: relative;
      padding: 2rem 1.5rem 2.5rem;
    }
  
    &__timeline-line {
      position: absolute;
      top: 2.25rem;
      left: 2.25rem;
      bottom: 2.5rem;
      width: 0.125rem;
      background-color: $gray-100;
    }
  }
  
  @media (max-width: 768px) {
    .work-experience {
      max-width: 80%;
      margin: 0px auto;
      margin-top: 24px;
      &__header {
        padding: 1.25rem;
      }
  
      &__timeline {
        padding: 1.5rem 1.25rem 2rem;
      }
  
      &__timeline-line {
        left: 1.25rem;
      }
    }
  }
  @media (min-width: 768px) {
    .work-experience {
      max-width: 720px;
      margin: 0px auto;
      margin-top: 24px;

    }
    
  }
  </style>