const imageInput = document.getElementById("imageInput");
const form = document.getElementById("uploadForm");

const resultDiv = document.getElementById("result");
const flowerName = document.getElementById("flowerName");
const scientificName = document.getElementById("scientificName");
const confidenceText = document.getElementById("confidence");

const originalImage = document.getElementById("originalImage");
const gradcamImage = document.getElementById("gradcamImage");

const loadingText = document.getElementById("loading");
const errorText = document.getElementById("error");

const API_URL = "http://127.0.0.1:8000/predict";

form.addEventListener("submit", async (e) => {
  e.preventDefault(); // 🔥 REQUIRED

  const file = imageInput.files[0];
  console.log("Selected file:", file);

  if (!file) {
    alert("Please select an image file first.");
    return;
  }

  // Reset UI
  resultDiv.classList.add("hidden");
  errorText.classList.add("hidden");
  loadingText.classList.remove("hidden");

  // Preview original image
  const reader = new FileReader();
  reader.onload = () => {
    originalImage.src = reader.result;
  };
  reader.readAsDataURL(file);

  // Build FormData PROPERLY
  const formData = new FormData(form); // <-- uses name="file"

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const data = await response.json();

    flowerName.textContent = data.prediction.english;
    scientificName.textContent = data.prediction.scientific;
    confidenceText.textContent =
      "Confidence: " + (data.prediction.confidence * 100).toFixed(2) + "%";

    // gradcamImage.src =  data.gradcam_image;

    loadingText.classList.add("hidden");
    resultDiv.classList.remove("hidden");
  } catch (err) {
    loadingText.classList.add("hidden");
    errorText.textContent = "Something went wrong. Please try again.";
    errorText.classList.remove("hidden");
    console.error(err);
  }
});
