/**
 * AI-Based Pneumonia Detection Using Attention-Enhanced CNN
 * Client-Side Application Logic (script.js)
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const fileNameLabel = document.getElementById('file-name-label');
    const fileSizeLabel = document.getElementById('file-size-label');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    const btnAnalyze = document.getElementById('btn-analyze');

    const loadingContainer = document.getElementById('loading-container');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const btnCloseError = document.getElementById('btn-close-error');

    const predictionCard = document.getElementById('prediction-card');
    const predictionBadge = document.getElementById('prediction-badge');
    const predictionText = document.getElementById('prediction-text');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const confidenceBar = document.getElementById('confidence-bar');
    const resultMessage = document.getElementById('result-message');
    const resultOriginalImg = document.getElementById('result-original-img');
    const resultHeatmapImg = document.getElementById('result-heatmap-img');
    const heatmapFrame = document.getElementById('heatmap-frame');
    const noHeatmapMessage = document.getElementById('no-heatmap-message');
    const btnReset = document.getElementById('btn-reset');
    const modelStatusBadge = document.getElementById('model-status-badge');

    // Configuration
    const ALLOWED_EXTENSIONS = ['image/jpeg', 'image/jpg', 'image/png'];
    const MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024; // 16 MB

    let selectedFile = null;
    let confidenceAnimationInterval = null;

    // Check system & model status on load
    checkModelStatus();

    // =========================================================================
    // File Input & Drag-and-Drop Handlers
    // =========================================================================

    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener('click', (e) => {
        if (!selectedFile && e.target !== browseBtn) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file);
        }
    });

    // Drag-over event listeners
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('drag-over');
        });
    });

    ['dragleave', 'dragend'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('drag-over');

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    btnRemoveFile.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUploadState();
    });

    btnCloseError.addEventListener('click', hideError);

    // =========================================================================
    // File Validation & Preview
    // =========================================================================

    function handleFileSelection(file) {
        hideError();

        // Validate File Type
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        const validExtensions = ['.jpg', '.jpeg', '.png'];
        const isValidType = ALLOWED_EXTENSIONS.includes(file.type) || validExtensions.includes(fileExt);

        if (!isValidType) {
            showError('Invalid file format. Please upload a chest X-ray in JPG, JPEG, or PNG format.');
            return;
        }

        // Validate File Size
        if (file.size > MAX_FILE_SIZE_BYTES) {
            showError('File is too large. Maximum supported image size is 16MB.');
            return;
        }

        selectedFile = file;

        // Render preview
        fileNameLabel.textContent = file.name;
        fileSizeLabel.textContent = formatBytes(file.size);

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            uploadPlaceholder.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden');
        };
        reader.onerror = () => {
            showError('Failed to read image file. Please try selecting another file.');
        };
        reader.readAsDataURL(file);
    }

    function formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function resetUploadState() {
        selectedFile = null;
        fileInput.value = '';
        previewImg.src = '';
        previewContainer.classList.add('hidden');
        uploadPlaceholder.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        loadingContainer.classList.add('hidden');
        hideError();
    }

    // =========================================================================
    // Analysis & Prediction Request
    // =========================================================================

    btnAnalyze.addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!selectedFile) {
            showError('Please upload a chest X-ray image first.');
            return;
        }

        // Switch to loading state
        hideError();
        dropZone.classList.add('hidden');
        loadingContainer.classList.remove('hidden');
        resultContainer.classList.add('hidden');

        // Prepare FormData
        const formData = new FormData();
        formData.append('file', selectedFile);
         const responseText = await response.text();

let data;

try {
    data = JSON.parse(responseText);
} catch (e) {
    console.error("Server returned non-JSON:", responseText);
    throw new Error(
        `Server returned an invalid response (${response.status}).`
    );
}

if (!response.ok) {
    throw new Error(data.error || 'Server error occurred during analysis.');
}

            const responseText= await response.text();
            let data;
            try {
            data = JSON.parse(responseText);
            } catch (e) {}
            ) {
                throw new Error('Failed to parse analysis results.');
            }

            if (!response.ok) {
                throw new Error(data.error || 'Server error occurred during analysis.');
            }

            // Display Results
            renderResults(data);
        } catch (err) {
            console.error('Prediction request failed:', err);
            showError(err.message || 'Failed to connect to the analysis server. Please check backend status.');
            dropZone.classList.remove('hidden');
        } finally {
            loadingContainer.classList.add('hidden');
        }
    });

    // =========================================================================
    // Render Results
    // =========================================================================

    function renderResults(data) {
        const isPneumonia = (data.prediction || '').toUpperCase() === 'PNEUMONIA';
        const confidenceVal = parseFloat(data.confidence) || 0.0;

        // Reset badge styles
        predictionBadge.classList.remove('pneumonia', 'normal');
        predictionCard.classList.remove('is-pneumonia', 'is-normal');

        if (isPneumonia) {
            predictionBadge.classList.add('pneumonia');
            predictionCard.classList.add('is-pneumonia');
            predictionText.textContent = 'PNEUMONIA';
        } else {
            predictionBadge.classList.add('normal');
            predictionCard.classList.add('is-normal');
            predictionText.textContent = 'NORMAL';
        }

        // Summary message
        resultMessage.textContent = data.message || (isPneumonia
            ? 'The Attention-Enhanced CNN detected characteristic infiltrates/opacities indicative of pneumonia.'
            : 'The Attention-Enhanced CNN indicates clear lung fields consistent with a normal radiograph.'
        );

        // Imagery display
        resultOriginalImg.src = data.original_url || previewImg.src;

        if (data.heatmap_url) {
            resultHeatmapImg.src = data.heatmap_url + '?t=' + new Date().getTime();
            heatmapFrame.classList.remove('hidden');
            noHeatmapMessage.classList.add('hidden');
        } else {
            heatmapFrame.classList.add('hidden');
            noHeatmapMessage.classList.remove('hidden');
        }

        // Show result container
        resultContainer.classList.remove('hidden');

        // Trigger smooth confidence animation
        animateConfidence(confidenceVal);

        // Scroll to results
        resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function animateConfidence(targetVal) {
        if (confidenceAnimationInterval) {
            clearInterval(confidenceAnimationInterval);
        }

        let current = 0;
        const step = targetVal / 40;
        confidencePercentage.textContent = '0.0%';
        confidenceBar.style.width = '0%';

        confidenceAnimationInterval = setInterval(() => {
            current += step;
            if (current >= targetVal) {
                current = targetVal;
                clearInterval(confidenceAnimationInterval);
            }
            confidencePercentage.textContent = current.toFixed(2) + '%';
            confidenceBar.style.width = current + '%';
        }, 20);
    }

    // Reset button
    btnReset.addEventListener('click', () => {
        dropZone.classList.remove('hidden');
        resetUploadState();
        dropZone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    });

    // =========================================================================
    // Error Handling
    // =========================================================================

    function showError(msg) {
        errorMessage.textContent = msg;
        errorContainer.classList.remove('hidden');
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }

    // =========================================================================
    // Check Backend Model Status
    // =========================================================================

    async function checkModelStatus() {
        try {
            const res = await fetch('/model-status');
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'ready' || data.status === 'loaded') {
                    modelStatusBadge.textContent = data.model_name ? `Model: ${data.model_name}` : 'AI Model Ready';
                } else if (data.status === 'initialized') {
                    modelStatusBadge.textContent = 'Model: Initialized (Baseline)';
                }
            }
        } catch (e) {
            // Keep default badge text if request fails
            console.log('Status endpoint unreachable:', e);
        }
    }
});
