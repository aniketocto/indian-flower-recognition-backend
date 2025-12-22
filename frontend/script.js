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

const scanLine = document.getElementById("scanLine");

const API_URL = "http://192.168.1.106:8000/predict";

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
    resultDiv.classList.remove("hidden"); // show image section
    scanLine.classList.remove("hidden"); // start scanning
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

    flowerName.textContent = "Flower Name: " + data.prediction.english;
    scientificName.textContent =
      "Scientific Name: " + data.prediction.scientific;
    confidenceText.textContent =
      "Confidence: " + (data.prediction.confidence * 100).toFixed(2) + "%";

    // gradcamImage.src =  data.gradcam_image;

    loadingText.classList.add("hidden");
    scanLine.classList.add("hidden"); // stop scanning

    flowerName.style.opacity = 0;
    scientificName.style.opacity = 0;
    confidenceText.style.opacity = 0;

    setTimeout(() => {
      flowerName.style.opacity = 1;
      scientificName.style.opacity = 1;
      confidenceText.style.opacity = 1;
    }, 200);
  } catch (err) {
    loadingText.classList.add("hidden");
    errorText.textContent = "Something went wrong. Please try again.";
    errorText.classList.remove("hidden");
    console.error(err);
  }
});

const fileNameText = document.getElementById("fileName");

imageInput.addEventListener("change", () => {
  if (imageInput.files.length > 0) {
    fileNameText.textContent = imageInput.files[0].name;
  } else {
    fileNameText.textContent = "No file chosen";
  }
});
