document.addEventListener('DOMContentLoaded', () => {
  const sections = document.querySelectorAll('.page-section');
  const navLinks = document.querySelectorAll('.nav-link');
 
  function showSection() {
    const hash = window.location.hash || '#home';
    sections.forEach(section => {
      section.classList.remove('active');
      if (`#${section.id}` === hash) {
        section.classList.add('active');
      }
    });
 
    navLinks.forEach(link => {
      link.classList.remove('active');
      if (link.getAttribute('href') === hash) {
        link.classList.add('active');
      }
    });
  }
 
  showSection();
  window.addEventListener('hashchange', showSection);
 
  navLinks.forEach(link => {
    link.addEventListener('click', () => {
      const hash = link.getAttribute('href');
      window.location.hash = hash;
    });
  });
 
  const uploadForm = document.getElementById('uploadForm');
  const fileInput = document.getElementById('fileInput');
  const uploadedImageContainer = document.getElementById('uploaded-image-container');
  const predictionResult = document.getElementById('prediction-result');
 
  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
 
    if (fileInput.files.length === 0) {
      predictionResult.textContent = 'Please select an image file.';
      predictionResult.classList.add('error');
      return;
    }
 
    // Display the uploaded image
    const file = fileInput.files[0];
    const imageUrl = URL.createObjectURL(file);
    uploadedImageContainer.innerHTML = `<img src="${imageUrl}" alt="Uploaded Nail Image" style="max-width: 300px; max-height: 300px;">`;
 
    const formData = new FormData();
    formData.append('file', file);
 
    predictionResult.textContent = 'Predicting...';
    predictionResult.classList.remove('error');
 
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData
      });
 
      const data = await response.json();
 
      if (data.error) {
        predictionResult.textContent = `Error: ${data.error}`;
        predictionResult.classList.add('error');
      } else {
        predictionResult.textContent = `Predicted Disease: ${data.disease_name} (Confidence: ${data.confidence}%)`;
        predictionResult.classList.remove('error');
      }
    } catch (error) {
      predictionResult.textContent = 'Failed to connect to the server.';
      predictionResult.classList.add('error');
    }
  });
});