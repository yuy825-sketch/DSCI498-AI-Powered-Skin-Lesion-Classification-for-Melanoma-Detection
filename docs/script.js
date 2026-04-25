const navToggle = document.querySelector(".nav-toggle");
const siteHeader = document.querySelector(".site-header");
const navLinks = [...document.querySelectorAll(".site-nav a")];
const sectionNavLinks = navLinks.filter((link) => link.getAttribute("href")?.startsWith("#"));
const pageNavLinks = navLinks.filter((link) => !link.getAttribute("href")?.startsWith("#"));
const revealNodes = [...document.querySelectorAll(".reveal")];
const modeButtons = [...document.querySelectorAll(".mode-button")];
const progressBar = document.getElementById("scroll-progress-bar");
const zoomableImages = [...document.querySelectorAll(".zoomable")];
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxCaption = document.getElementById("lightbox-caption");
const lightboxClose = document.getElementById("lightbox-close");
const modeData = {
  accuracy: {
    kicker: "Highest multiclass accuracy",
    title: "EffNet-B2 @ 260px selected by validation accuracy",
    description:
      "This mode gives the strongest overall classifier on the fixed test split, but its top-1 melanoma recall is much lower than the sensitivity-oriented setup.",
    accuracy: "0.8614",
    macro: "0.7386",
    mel: "0.5403",
    points: [
      "Best final multiclass performance on the test split.",
      "High confidence predictions need calibration diagnostics to avoid false trust.",
      "Useful as the headline model for the paper, README, and report summary."
    ],
    image: "assets/confusion_acc.png",
    alt: "Accuracy-focused confusion matrix",
    caption:
      "Confusion matrix for the best accuracy-focused run. Strong `nv` performance helps overall accuracy, but melanoma misses remain visible."
  },
  sensitivity: {
    kicker: "Highest top-1 melanoma recall",
    title: "EffNet-B2 @ 260px with weighted sampling and melanoma-aware selection",
    description:
      "This mode pushes the system toward screening behavior: many more melanoma cases are recovered, but overall multiclass utility drops sharply.",
    accuracy: "0.5374",
    macro: "0.5666",
    mel: "0.8548",
    points: [
      "Recovers substantially more melanoma cases under top-1 classification.",
      "Pays for recall with strong confusion on non-melanoma classes, especially `nv`.",
      "Best presented as a safety-oriented operating mode, not as the single best classifier."
    ],
    image: "assets/confusion_sens.png",
    alt: "Sensitivity-first confusion matrix",
    caption:
      "Confusion matrix for the sensitivity-first run. Melanoma recall improves, but false positives spread across several non-melanoma classes."
  },
  threshold: {
    kicker: "Decision threshold instead of retraining",
    title: "Thresholding P(mel) turns a multiclass model into a screening score",
    description:
      "The tuned EffNet-B2 model can be reframed as a one-vs-rest melanoma detector by sweeping a fixed threshold over P(mel), making the precision-recall trade-off explicit.",
    accuracy: "0.7697",
    macro: "0.6958",
    mel: "0.855 recall @ t=0.13",
    points: [
      "Keeps the tuned backbone but changes the decision policy instead of the training objective.",
      "Best precision under the recall >= 0.85 constraint is approximately 0.322.",
      "Useful for explaining why operating-point analysis belongs in medical-style evaluation."
    ],
    image: "assets/mel_threshold_curve.png",
    alt: "Melanoma threshold curve",
    caption:
      "Precision and recall versus threshold for melanoma one-vs-rest detection. The useful high-recall regime comes with a visible precision cost."
  }
};

if (navToggle && siteHeader) {
  navToggle.addEventListener("click", () => {
    const expanded = navToggle.getAttribute("aria-expanded") === "true";
    navToggle.setAttribute("aria-expanded", String(!expanded));
    siteHeader.classList.toggle("nav-open");
  });
}

const sectionIds = sectionNavLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const onScroll = () => {
  const current = sectionIds.find((section) => {
    const rect = section.getBoundingClientRect();
    return rect.top <= 140 && rect.bottom >= 140;
  });

  sectionNavLinks.forEach((link) => {
    const isActive = current && link.getAttribute("href") === `#${current.id}`;
    link.classList.toggle("active", Boolean(isActive));
  });

  if (progressBar) {
    const scrollTop = window.scrollY;
    const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
    const percent = scrollHeight > 0 ? (scrollTop / scrollHeight) * 100 : 0;
    progressBar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
  }
};

window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.18 }
);

revealNodes.forEach((node) => observer.observe(node));

const assignMode = (modeKey) => {
  const mode = modeData[modeKey];
  if (!mode) {
    return;
  }

  document.getElementById("mode-kicker").textContent = mode.kicker;
  document.getElementById("mode-title").textContent = mode.title;
  document.getElementById("mode-description").textContent = mode.description;
  document.getElementById("metric-accuracy").textContent = mode.accuracy;
  document.getElementById("metric-macro").textContent = mode.macro;
  document.getElementById("metric-mel").textContent = mode.mel;

  const image = document.getElementById("mode-image");
  image.src = mode.image;
  image.alt = mode.alt;
  document.getElementById("mode-caption").textContent = mode.caption;

  const points = document.getElementById("mode-points");
  points.innerHTML = mode.points.map((point) => `<li>${point}</li>`).join("");

  modeButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.mode === modeKey);
  });
};

modeButtons.forEach((button) => {
  button.addEventListener("click", () => assignMode(button.dataset.mode));
});

assignMode("accuracy");

pageNavLinks.forEach((link) => {
  const href = link.getAttribute("href");
  if (!href) {
    return;
  }

  const currentPage = window.location.pathname.split("/").pop() || "index.html";
  link.classList.toggle("active", href === currentPage);
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    if (siteHeader?.classList.contains("nav-open")) {
      siteHeader.classList.remove("nav-open");
      navToggle?.setAttribute("aria-expanded", "false");
    }
  });
});

const closeLightbox = () => {
  if (!lightbox) {
    return;
  }

  lightbox.classList.remove("is-open");
  lightbox.setAttribute("aria-hidden", "true");
  document.body.classList.remove("lightbox-open");
};

zoomableImages.forEach((image) => {
  image.addEventListener("click", () => {
    if (!lightbox || !lightboxImage || !lightboxCaption) {
      return;
    }

    lightboxImage.src = image.currentSrc || image.src;
    lightboxImage.alt = image.alt;
    const captionSource = image.closest("figure")?.querySelector("figcaption");
    lightboxCaption.textContent = captionSource?.textContent?.trim() || image.alt;
    lightbox.classList.add("is-open");
    lightbox.setAttribute("aria-hidden", "false");
    document.body.classList.add("lightbox-open");
  });
});

lightboxClose?.addEventListener("click", closeLightbox);

lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) {
    closeLightbox();
  }
});

window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeLightbox();
  }
});
