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
const exampleCards = [...document.querySelectorAll(".example-card")];
const exampleLoading = document.getElementById("example-loading");
const exampleLoadingText = document.getElementById("example-loading-text");
const exampleStatus = document.getElementById("example-status");
const exampleResult = document.getElementById("example-result");
const exampleImage = document.getElementById("example-image");
const exampleCaption = document.getElementById("example-caption");
const exampleTitle = document.getElementById("example-title");
const exampleDescription = document.getElementById("example-description");
const exampleBars = document.getElementById("example-bars");
const exampleNote = document.getElementById("example-note");
let exampleTimer = null;
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
const exampleData = {
  mel_missed: {
    label: "False-negative melanoma",
    title: "Melanoma missed under argmax",
    description:
      "Truth is melanoma, but the baseline model predicts `vasc` with high confidence. This is a high-risk failure mode and a clean justification for sensitivity-oriented evaluation.",
    image: "assets/example_mel_missed.png",
    caption:
      "ISIC_0025964 overlay. The stored figure combines the selected lesion view with a Grad-CAM style attention overlay.",
    note:
      "Top-1 prediction is wrong even though the attention appears lesion-focused. Explanation helps inspect behavior, but it does not guarantee safe classification.",
    probs: [
      { label: "vasc", value: 0.5477 },
      { label: "nv", value: 0.368 },
      { label: "mel", value: 0.074 }
    ]
  },
  nv_correct: {
    label: "Correct benign nevus",
    title: "Correct `nv`, but melanoma score is still non-trivial",
    description:
      "This benign nevus is classified correctly, yet the melanoma probability remains relatively high. It illustrates why false positives can increase when the site emphasizes sensitivity.",
    image: "assets/example_nv_correct.png",
    caption:
      "ISIC_0024698 overlay. This prepared case shows a correct common-class prediction with a meaningful melanoma score in the top-3.",
    note:
      "This kind of case is useful when explaining why benign pigmented lesions can become false positives under high-sensitivity operating points.",
    probs: [
      { label: "nv", value: 0.6598 },
      { label: "mel", value: 0.3281 },
      { label: "bkl", value: 0.0111 }
    ]
  },
  bcc_correct: {
    label: "Correct basal cell carcinoma",
    title: "Strong non-melanoma separation",
    description:
      "This basal cell carcinoma case is predicted correctly with very high confidence. It acts as a sanity-check example showing that not every lesion type is equally ambiguous for the model.",
    image: "assets/example_bcc_correct.png",
    caption:
      "ISIC_0028155 overlay. The prepared output reflects a stable non-melanoma decision with negligible melanoma probability.",
    note:
      "Cases like this show where the classifier is confident and stable, in contrast to the much harder melanoma-versus-nevus boundary.",
    probs: [
      { label: "bcc", value: 0.999 },
      { label: "mel", value: 0.0004 },
      { label: "bkl", value: 0.0003 }
    ]
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

const renderExample = (exampleKey) => {
  const example = exampleData[exampleKey];
  if (!example || !exampleImage || !exampleBars) {
    return;
  }

  exampleImage.src = example.image;
  exampleImage.alt = example.label;
  exampleCaption.textContent = example.caption;
  exampleTitle.textContent = example.title;
  exampleDescription.textContent = example.description;
  exampleNote.textContent = example.note;
  exampleStatus.textContent = `${example.label} selected. Stored output shown below.`;
  exampleBars.innerHTML = example.probs
    .map(
      (prob) => `
        <div class="prob-row">
          <div class="prob-head">
            <span>${prob.label}</span>
            <strong>${prob.value.toFixed(4)}</strong>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${Math.max(prob.value * 100, prob.value > 0 ? 2 : 0)}%"></div>
          </div>
        </div>
      `
    )
    .join("");

  exampleCards.forEach((card) => {
    const active = card.dataset.example === exampleKey;
    card.classList.toggle("is-active", active);
    card.setAttribute("aria-pressed", String(active));
  });
};

const playExample = (exampleKey) => {
  if (!exampleLoading || !exampleResult || !exampleLoadingText || !exampleStatus) {
    return;
  }

  if (exampleTimer) {
    clearTimeout(exampleTimer);
  }

  exampleStatus.textContent = `Selected ${exampleData[exampleKey].label}. Replaying fixed prediction flow...`;
  exampleLoading.hidden = false;
  exampleResult.hidden = true;
  exampleLoadingText.textContent = "Loading precomputed prediction...";

  exampleTimer = window.setTimeout(() => {
    renderExample(exampleKey);
    exampleLoading.hidden = true;
    exampleResult.hidden = false;
  }, 1600);
};

exampleCards.forEach((card) => {
  card.addEventListener("click", () => playExample(card.dataset.example));
});

if (exampleCards.length > 0) {
  renderExample("mel_missed");
}
