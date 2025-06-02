// Coral Reef Health Detection - Main JavaScript

// Global variables
let uploadedFile = null;
let analysisResults = null;

// DOM elements
const fileInput = document.getElementById('fileInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');

// Initialize event listeners
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
});

function initializeEventListeners() {
    if (fileInput) fileInput.addEventListener('change', handleFileSelect);
    if (uploadArea) {
        uploadArea.addEventListener('click', () => fileInput && fileInput.click());
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
    }
    if (analyzeBtn) analyzeBtn.addEventListener('click', analyzeImage);
    
    document.addEventListener('dragover', e => e.preventDefault());
    document.addEventListener('drop', e => e.preventDefault());
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) processFile(file);
}

function handleDragOver(event) {
    event.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = event.dataTransfer.files;
    if (files.length > 0) processFile(files[0]);
}

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    
    uploadedFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        if (imagePreview) imagePreview.src = e.target.result;
        if (previewSection) previewSection.style.display = 'block';
        if (resultsSection) resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!uploadedFile) {
        alert('Please select an image first.');
        return;
    }
    
    if (loadingSection) loadingSection.style.display = 'block';
    if (previewSection) previewSection.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', uploadedFile);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const results = await response.json();
        if (results.error) throw new Error(results.error);
        
        displayResults(results);
    } catch (error) {
        alert('Analysis failed: ' + error.message);
        if (loadingSection) loadingSection.style.display = 'none';
        if (previewSection) previewSection.style.display = 'block';
    }
}

function displayResults(results) {
    if (loadingSection) loadingSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'block';
    
    // Update health badge
    const healthBadge = document.getElementById('healthBadge');
    if (healthBadge) {
        healthBadge.textContent = results.overall_health;
        healthBadge.className = `health-badge ${results.overall_health}`;
    }
    
    // Update health score
    const healthScoreNumber = document.getElementById('healthScoreNumber');
    if (healthScoreNumber) {
        healthScoreNumber.textContent = Math.round(results.health_score);
    }
    
    // Update description
    const healthDescription = document.getElementById('healthDescription');
    if (healthDescription) {
        healthDescription.textContent = results.description;
    }
    
    // Update confidence
    const confidenceValue = document.getElementById('confidenceValue');
    if (confidenceValue) {
        confidenceValue.textContent = Math.round(results.overall_confidence * 100) + '%';
    }
    
    // Update patch count
    const patchCount = document.getElementById('patchCount');
    if (patchCount) {
        patchCount.textContent = results.total_patches_analyzed;
    }
    
    // Update recommendations
    const recommendationsList = document.getElementById('recommendationsList');
    if (recommendationsList && results.recommendations) {
        recommendationsList.innerHTML = '';
        results.recommendations.forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
    }
}

function resetAnalysis() {
    uploadedFile = null;
    if (fileInput) fileInput.value = '';
    if (previewSection) previewSection.style.display = 'none';
    if (loadingSection) loadingSection.style.display = 'none';
    if (resultsSection) resultsSection.style.display = 'none';
}

// Global functions for HTML
window.resetAnalysis = resetAnalysis;

console.log('Coral Reef Health Detection - JavaScript Loaded');