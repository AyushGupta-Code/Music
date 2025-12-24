const form = document.getElementById("generator-form");
const submitBtn = document.getElementById("submit-btn");
const progressSection = document.getElementById("progress");
const progressFill = document.querySelector(".progress-fill");
const resultSection = document.getElementById("result");
const audioPlayer = document.getElementById("audio-player");
const resetBtn = document.getElementById("reset-btn");

let progressTimer;

const setHidden = (element, hidden) => {
  if (hidden) {
    element.classList.add("hidden");
  } else {
    element.classList.remove("hidden");
  }
};

const startProgress = () => {
  clearInterval(progressTimer);
  let progress = 0;
  progressFill.style.width = "0%";
  setHidden(progressSection, false);
  submitBtn.disabled = true;

  progressTimer = setInterval(() => {
    progress = Math.min(progress + Math.random() * 15, 95);
    progressFill.style.width = `${progress}%`;
  }, 400);
};

const finishProgress = () => {
  clearInterval(progressTimer);
  progressFill.style.width = "100%";
  submitBtn.disabled = false;
  setTimeout(() => setHidden(progressSection, true), 300);
};

form?.addEventListener("submit", async (event) => {
  event.preventDefault();
  setHidden(resultSection, true);
  startProgress();

  try {
    const response = await fetch("/generate", {
      method: "POST",
      body: new FormData(form),
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(message || "Failed to generate audio");
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    audioPlayer.src = url;
    setHidden(resultSection, false);
  } catch (error) {
    alert(error.message || "Something went wrong while generating audio.");
  } finally {
    finishProgress();
  }
});

resetBtn?.addEventListener("click", () => {
  form.reset();
  setHidden(resultSection, true);
});
