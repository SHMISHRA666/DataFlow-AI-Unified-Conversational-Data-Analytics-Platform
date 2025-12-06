// Global variables
let currentSessionId = null;
let currentAnalysisType = null; // not used now, kept for compatibility
let fileInfo = null;
// Tracks if a conversation is already active; first Ask after Start New Conversation will start it
let conversationActive = false;
let conversationHistory = []; // Stores the full conversation

// Dedicated handler to open file dialog from upload area
function onUploadAreaClick() {
    const input = document.getElementById('file-input');
    if (input) input.click();
}

// Initialize the application
DocumentReady = false;
document.addEventListener('DOMContentLoaded', function() {
    DocumentReady = true;
    initializeApp();
    // Make sure the upload section is visible when starting a new conversation
    updateDatasetUI('upload');
    
    // If upload area content is missing (e.g., after previous load), render it
    renderUploadArea();
});

function initializeApp() {
    // Ensure upload area has proper content before wiring listeners
    renderUploadArea();
    setupFileUpload();
    setupDragAndDrop();
    setupDatasetSelector();
    loadHistory();
}

// Ensure the upload area contains the clickable content and input
function renderUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    if (!uploadArea) return;

    // If the area already has an input, assume it's fine
    const existingInput = uploadArea.querySelector('#file-input');
    if (!existingInput) {
        uploadArea.innerHTML = `
            <div class="upload-content">
                <i class="fas fa-cloud-upload-alt upload-icon"></i>
                <h3>Drag & Drop your file here</h3>
                <p>or click to browse files</p>
                <p class="small">Supports CSV, JSON, Excel, HTML, and PDF files</p>
                <input type="file" id="file-input" accept=".csv,.json,.xlsx,.xls,.html,.htm,.pdf" style="display: none;">
            </div>
        `;
    }

    // Make sure it's visible
    uploadArea.style.display = 'flex';
    uploadArea.style.visibility = 'visible';
    uploadArea.style.opacity = '1';

    // Rebind file input change listener if needed
    setupFileUpload();

    // Rebind click to trigger the hidden file input
    // Moved to setupDragAndDrop to avoid duplicate bindings
}

function togglePreChat(show) {
    const pre = document.getElementById('pre-chat-container');
    if (!pre) return;
    pre.classList.toggle('hidden', !show);
}

function startNewConversation() {
    try {
        // Reset state
        currentSessionId = null;
        currentAnalysisType = null;
        fileInfo = null;
        conversationActive = false;
        
        // Reset file input
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';
        
        // Hide file info
        const fileInfoDiv = document.getElementById('file-info');
        if (fileInfoDiv) fileInfoDiv.classList.add('hidden');
        
        // Clear question input
        const questionInput = document.getElementById('question-input');
        if (questionInput) questionInput.value = '';
        
        // Clear chat history
        const chatHistory = document.getElementById('chat-history');
        if (chatHistory) chatHistory.innerHTML = '';
        
        // Reset dataset selector
        const datasetSelect = document.getElementById('dataset-select');
        if (datasetSelect) {
            datasetSelect.value = 'upload';
            updateDatasetUI('upload');
        }
        
        // Show pre-chat container
        const preChatContainer = document.getElementById('pre-chat-container');
        if (preChatContainer) {
            preChatContainer.style.display = 'block';
            preChatContainer.classList.remove('hidden');
        }
        
        // Hide chat container
        const chatContainer = document.getElementById('chat-container');
        if (chatContainer) {
            chatContainer.style.display = 'none';
        }
        
        // Ensure upload section and area are visible and correctly rendered
        const uploadSection = document.getElementById('upload-section');
        if (uploadSection) {
            uploadSection.style.display = 'block';
            uploadSection.style.visibility = 'visible';
            uploadSection.style.opacity = '1';
        }
        renderUploadArea();
        
        // Force a reflow to ensure styles are applied
        if (preChatContainer) {
            preChatContainer.offsetHeight;
        }
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
    } catch (error) {
        console.error('Error in startNewConversation:', error);
        showError('Failed to start a new conversation. Please refresh the page.');
    }
}

// Dataset selector
function setupDatasetSelector() {
    const select = document.getElementById('dataset-select');
    if (!select) return;
    const update = () => updateDatasetUI(select.value);
    select.addEventListener('change', update);
    update();
}

function updateDatasetUI(value) {
    const uploadSection = document.getElementById('upload-section');
    if (!uploadSection) return;
    if (value === 'upload') {
        uploadSection.style.display = '';
    } else {
        // Hide upload for built-in datasets (backend to be added later)
        uploadSection.style.display = 'none';
        currentSessionId = null;
        fileInfo = null;
        const fi = document.getElementById('file-info');
        if (fi) fi.classList.add('hidden');
    }
}

// File Upload Functionality
function setupFileUpload() {
    const fileInput = document.getElementById('file-input');
    if (!fileInput) return;
    // Avoid duplicate listeners on re-render
    fileInput.removeEventListener('change', handleFileSelect);
    fileInput.addEventListener('change', handleFileSelect);
}

function setupDragAndDrop() {
    const uploadArea = document.getElementById('upload-area');
    if (!uploadArea) return;

    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Ensure only a single click listener exists
    uploadArea.removeEventListener('click', onUploadAreaClick);
    uploadArea.addEventListener('click', onUploadAreaClick);
}

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    const allowedExtensions = ['.csv', '.json', '.xlsx', '.xls', '.html', '.htm', '.pdf'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(fileExtension)) {
        showError('Please upload a CSV, JSON, Excel, HTML, or PDF file.');
        return;
    }
    
    // Upload file
    uploadFile(file);
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading('upload-area', 'Uploading file...');
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentSessionId = data.session_id;
            fileInfo = data.file_info;
            
            // Show chat container and hide pre-chat
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) chatContainer.style.display = 'flex';
            togglePreChat(false);
            
            // Display file info
            displayFileInfo(data);
            
            // Scroll to chat input
            const askSection = document.getElementById('ask-section');
            if (askSection) askSection.scrollIntoView({ behavior: 'smooth' });
        } else {
            showError(data.error || 'Upload failed');
        }
    })
    .catch(error => {
        console.error('Upload error:', error);
        showError('Upload failed. Please try again.');
    })
    .finally(() => {
        hideLoading('upload-area');
    });
}

function displayFileInfo(data) {
    const fileInfoDiv = document.getElementById('file-info');
    const fileDetailsDiv = document.getElementById('file-details');
    if (!fileInfoDiv || !fileDetailsDiv) return;
    
    fileDetailsDiv.innerHTML = `
        <div class="file-detail">
            <h4>File Name</h4>
            <p>${data.original_filename}</p>
        </div>
        <div class="file-detail">
            <h4>Records</h4>
            <p>${data.file_info.shape[0]} rows</p>
        </div>
        <div class="file-detail">
            <h4>Columns</h4>
            <p>${data.file_info.shape[1]} columns</p>
        </div>
        <div class="file-detail">
            <h4>Column Names</h4>
            <p>${(data.file_info.columns || []).join(', ')}</p>
        </div>
    `;
    
    fileInfoDiv.classList.remove('hidden');
}

// Ask Question Flow
function askQuestion() {
    const ds = document.getElementById('dataset-select');
    if (ds && ds.value !== 'upload') {
        showError('Built-in datasets are coming soon. Please select "Upload your own data" for now.');
        return;
    }

    const questionInput = document.getElementById('question-input');
    const question = (questionInput?.value || '').trim();
    
    if (!currentSessionId) {
        showError('Please upload a file first.');
        return;
    }
    if (!question) {
        showError('Please enter a question.');
        return;
    }
    
    // Add user question to conversation history
    addMessageToChat('user', question);
    
    // Clear input
    if (questionInput) questionInput.value = '';
    
    // Save to history if this is the first question of a new conversation
    if (!conversationActive) {
        saveHistory(question);
        conversationActive = true;
    }
    
    // Show loading indicator
    addMessageToChat('assistant', 'Processing your question...', true);
    
    fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: currentSessionId,
            question: question
        })
    })
    .then(res => res.json())
    .then(data => {
        // Remove loading indicator
        const loadingMessages = document.querySelectorAll('.message.loading');
        loadingMessages.forEach(el => el.remove());
        
        if (!data.success) {
            addMessageToChat('error', data.error || 'Classification failed');
            return;
        }
        
        // Hide pre-chat (welcome + dropdown + upload)
        togglePreChat(false);
        
        
        const finalAnswer = data.final_answer_text || '';
        const artifacts = data.artifacts || {};
        let artifactUrl = '';
        if (artifacts.preferred_entry && (artifacts.preferred_entry.public_url || artifacts.preferred_entry.relative)) {
            artifactUrl = artifacts.preferred_entry.public_url || (`/${artifacts.preferred_entry.relative}`);
        } else if (artifacts.files && artifacts.files.html && artifacts.files.html.length > 0) {
            const first = artifacts.files.html[0];
            artifactUrl = first.public_url || (`/${first.relative}`);
        }

        let resultsHtml = '';
        if (finalAnswer) {
            resultsHtml += `
                <div class="json-response">
                    <div class="json-title">Answer</div>
                    <div class="json-output">${escapeHtml(finalAnswer)}</div>
                </div>
            `;
        }
        if (artifactUrl) {
            resultsHtml += `
                <div class="json-response">
                    <div class="json-title">Report</div>
                    <a href="${artifactUrl}" target="_blank" rel="noopener" class="btn-primary">Open Generated Report</a>
                </div>
            `;
        }
        if (resultsHtml) {
            addMessageToChat('assistant', resultsHtml);
        }
        
    })
    .catch(err => {
        console.error('Ask error:', err);
        addMessageToChat('error', 'Failed to process your question. Please try again.');
    });
}

function addMessageToChat(role, content, isLoading = false) {
    const chatHistory = document.getElementById('chat-history') || createChatHistoryElement();
    if (!chatHistory) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role} ${isLoading ? 'loading' : ''}`;
    
    if (isLoading) {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">
                ${content}
            </div>
        `;
    }
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    
    // Add to conversation history
    if (!isLoading) {
        conversationHistory.push({ role, content });
    }
}

function createChatHistoryElement() {
    const chatContainer = document.getElementById('chat-container');
    if (!chatContainer) return null;
    
    // Create chat history element if it doesn't exist
    let chatHistory = document.getElementById('chat-history');
    if (!chatHistory) {
        chatHistory = document.createElement('div');
        chatHistory.id = 'chat-history';
        chatContainer.insertBefore(chatHistory, document.getElementById('ask-section'));
    }
    
    return chatHistory;
}

// History helpers
function loadHistory() {
    fetch('/history')
        .then(r => r.json())
        .then(data => {
            if (data && data.success) {
                renderHistory(data.history || []);
            }
        })
        .catch(() => {});
}

function saveHistory(question) {
    fetch('/history', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
    })
    .then(() => loadHistory())
    .catch(() => {});
}

function renderHistory(items) {
    const list = document.getElementById('history-list');
    if (!list) return;
    list.innerHTML = '';
    if (!items.length) {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.textContent = 'No questions yet.';
        list.appendChild(li);
        return;
    }
    items.forEach(entry => {
        const li = document.createElement('li');
        li.className = 'history-item';
        li.innerHTML = `
            <div>${escapeHtml(entry.question || '')}</div>
            <time>${formatTime(entry.timestamp)}</time>
        `;
        list.appendChild(li);
    });
}

function formatTime(ts) {
    try { return new Date(ts).toLocaleString(); } catch { return ts || ''; }
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe.toString()
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Utility Functions
function showLoading(containerId, message = 'Loading...') {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Make sure container can host an overlay
    const prevPosition = window.getComputedStyle(container).position;
    if (prevPosition === 'static' || !prevPosition) {
        container.style.position = 'relative';
    }

    // Do NOT replace innerHTML; add an overlay instead
    let overlay = container.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-box">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
        container.appendChild(overlay);
    }
}

function hideLoading(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const overlay = container.querySelector('.loading-overlay');
    if (overlay) overlay.remove();
}

function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: #dc3545;
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        max-width: 360px;
        font-size: 14px;
    `;
    errorDiv.innerHTML = `
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
            <span><i class="fas fa-exclamation-triangle"></i> ${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: white; cursor: pointer; font-size: 18px;">×</button>
        </div>
    `;
    
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 4000);
}
