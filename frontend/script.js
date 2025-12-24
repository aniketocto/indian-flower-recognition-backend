const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");
const predictBtn = document.getElementById("predictBtn");
const imagePreview = document.getElementById("imagePreview");
const previewImg = document.getElementById("previewImg");
const scanOverlay = document.getElementById("scanOverlay");
const analyzingText = document.getElementById("analyzingText");
const resultSection = document.getElementById("resultSection");
const flowerName = document.getElementById("flowerName");
const scientificName = document.getElementById("scientificName");
const confidenceText = document.getElementById("confidenceText");
const progressFill = document.getElementById("progressFill");

// 🔥 YOUR REAL API
const API_URL = "http://192.168.1.106:8000/predict";

/* ---------------- FILE SELECT ---------------- */
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  fileName.textContent = file.name;
  predictBtn.disabled = false;

  const reader = new FileReader();
  reader.onload = (event) => {
    previewImg.src = event.target.result;
    imagePreview.classList.remove("hidden");

    // reset states
    resultSection.classList.add("hidden");
    scanOverlay.classList.add("hidden");
    analyzingText.classList.add("hidden");
    progressFill.style.width = "0%";
  };
  reader.readAsDataURL(file);
});

/* ---------------- PREDICT ---------------- */
predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) return;

  // UI → analyzing
  predictBtn.disabled = true;
  predictBtn.classList.add("analyzing");
  predictBtn.textContent = "🔮🔮🔮 Analyzing...";
  resultSection.classList.add("hidden");
  scanOverlay.classList.remove("hidden");
  analyzingText.classList.remove("hidden");
  progressFill.style.width = "0%";

  const formData = new FormData();
  formData.append("file", file); // ⚠️ matches backend key

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();

    /*
      Expected backend response:
      {
        prediction: {
          english: "Dahlia",
          scientific: "Dahlia pinnata",
          confidence: 0.947
        }
      }
    */

    const confidencePercent = (data.prediction.confidence * 100).toFixed(2);

    // Stop scanning
    scanOverlay.classList.add("hidden");
    analyzingText.classList.add("hidden");

    // Update result
    flowerName.textContent = data.prediction.english;
    scientificName.textContent = `Scientific name: ${data.prediction.scientific}`;
    confidenceText.textContent = `Confidence: ${confidencePercent}%`;

    resultSection.classList.remove("hidden");

    setTimeout(() => {
      progressFill.style.width = confidencePercent + "%";
    }, 150);
  } catch (err) {
    alert("❌ Something went wrong. Please try again.");
    console.error(err);
  } finally {
    predictBtn.disabled = false;
    predictBtn.classList.remove("analyzing");
    predictBtn.textContent = "🔮 Predict";
  }
});
