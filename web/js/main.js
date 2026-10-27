/**
 * SignSense Main Router & Shell Application
 */
import { api } from './api.js';

// Route Definitions
const ROUTES = {
  '/': { title: 'SignSense: Traffic Sign Recognition', module: () => import('./pages/home.js') },
  '/recognise': { title: 'Recognise | SignSense', module: () => import('./pages/recognise.js') },
  '/compare': { title: 'Compare Models | SignSense', module: () => import('./pages/compare.js') },
  '/attack': { title: 'Attack Lab | SignSense', module: () => import('./pages/attack.js') },
  '/report': { title: 'Model Report | SignSense', module: () => import('./pages/report.js') },
};

let currentPageModule = null;
let currentPath = null;

// DOM Elements
const contentArea = document.getElementById('content-area');
const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
const drawerLinks = document.querySelectorAll('.mobile-drawer .nav-link');
const navContainer = document.querySelector('.navbar-nav');
const laneMarker = document.querySelector('.lane-marker');
const statusPill = document.getElementById('status-pill');
const statusText = document.getElementById('status-text');
const hamburgerBtn = document.getElementById('hamburger-btn');
const mobileDrawer = document.getElementById('mobile-drawer');

/**
 * Slide amber lane marker to active link
 */
function updateLaneMarker(activeLink) {
  if (!laneMarker || !navContainer || !activeLink) {
    if (laneMarker) laneMarker.style.opacity = '0';
    return;
  }
  const navRect = navContainer.getBoundingClientRect();
  const linkRect = activeLink.getBoundingClientRect();
  const offsetLeft = linkRect.left - navRect.left;
  const width = linkRect.width;

  laneMarker.style.opacity = '1';
  laneMarker.style.width = `${width}px`;
  laneMarker.style.transform = `translateX(${offsetLeft}px)`;
}

/**
 * Highlight active link in Desktop & Mobile Nav
 */
function updateActiveNav(path) {
  const normalizedPath = path.startsWith('/report') ? '/report' : path;
  let activeElement = null;

  navLinks.forEach((link) => {
    const href = link.getAttribute('href');
    const linkPath = href.replace(/^#/, '');
    const isActive = linkPath === normalizedPath || (linkPath !== '/' && normalizedPath.startsWith(linkPath));
    link.classList.toggle('active', isActive);
    link.setAttribute('aria-current', isActive ? 'page' : 'false');
    if (isActive) activeElement = link;
  });

  drawerLinks.forEach((link) => {
    const href = link.getAttribute('href');
    const linkPath = href.replace(/^#/, '');
    const isActive = linkPath === normalizedPath || (linkPath !== '/' && normalizedPath.startsWith(linkPath));
    link.classList.toggle('active', isActive);
    link.setAttribute('aria-current', isActive ? 'page' : 'false');
  });

  updateLaneMarker(activeElement);
}

/**
 * Navigate to Hash Route
 */
async function navigate() {
  const hash = window.location.hash || '#/';
  let path = hash.replace(/^#/, '');
  if (!path.startsWith('/')) path = '/' + path;

  // Cleanup prior page
  if (currentPageModule && typeof currentPageModule.destroy === 'function') {
    try {
      currentPageModule.destroy();
    } catch (e) {
      console.error('Error during page teardown:', e);
    }
  }

  // Mobile drawer close on navigate
  if (mobileDrawer && mobileDrawer.classList.contains('open')) {
    mobileDrawer.classList.remove('open');
    if (hamburgerBtn) hamburgerBtn.setAttribute('aria-expanded', 'false');
  }

  currentPath = path;
  updateActiveNav(path);

  // Match root or sub-routes
  let routeKey = path;
  if (!ROUTES[routeKey] && path.startsWith('/report')) {
    routeKey = '/report';
  }

  const route = ROUTES[routeKey];

  // Quick Fade Transition
  contentArea.classList.add('fade-out');

  setTimeout(async () => {
    window.scrollTo({ top: 0, behavior: 'instant' });

    if (route) {
      document.title = route.title;
      try {
        const mod = await route.module();
        currentPageModule = mod;
        contentArea.innerHTML = '';
        mod.render(contentArea);
      } catch (err) {
        console.error('Failed to load page module:', err);
        renderErrorState('Failed to load module');
      }
    } else {
      // 404 Route
      document.title = 'Page Not Found | SignSense';
      render404();
    }

    contentArea.classList.remove('fade-out');
  }, 150);
}

function render404() {
  contentArea.innerHTML = `
    <div class="empty-state" style="padding: 64px 20px;">
      <div style="font-size: 48px;">🚫</div>
      <h2 style="font-size: var(--text-2xl); font-family: var(--font-heading); margin-top: 12px;">Page Not Found</h2>
      <p style="color: var(--muted); max-width: 400px; margin-bottom: 20px;">
        The requested URL path does not match any valid SignSense navigation routes.
      </p>
      <a href="#/" class="btn btn-primary">Back to Home</a>
    </div>
  `;
}

function renderErrorState(msg) {
  contentArea.innerHTML = `
    <div class="error-banner" style="margin-top: 24px;">
      ${msg}. Please check the server connection and try again.
    </div>
  `;
}

/**
 * Health Pill Monitor
 */
async function updateHealthStatus() {
  try {
    const data = await api.getHealth();
    if (data.status === 'ok') {
      statusPill.classList.remove('offline');
      statusText.textContent = `${data.models.length} models ready`;
    } else {
      throw new Error('Unhealthy status');
    }
  } catch (err) {
    statusPill.classList.add('offline');
    statusText.textContent = 'Backend offline';
  }
}

// Expose on window for runtime testing & health triggers
window.updateHealthStatus = updateHealthStatus;

// Mobile Hamburger Toggle
if (hamburgerBtn && mobileDrawer) {
  hamburgerBtn.addEventListener('click', () => {
    const isOpen = mobileDrawer.classList.toggle('open');
    hamburgerBtn.setAttribute('aria-expanded', isOpen ? 'true' : 'false');
  });
}

// Window events
window.addEventListener('hashchange', navigate);
window.addEventListener('resize', () => {
  const activeLink = document.querySelector('.navbar-nav .nav-link.active');
  updateLaneMarker(activeLink);
});

// Initial boot
window.addEventListener('DOMContentLoaded', () => {
  navigate();
  updateHealthStatus();
  setInterval(updateHealthStatus, 10000);
});
