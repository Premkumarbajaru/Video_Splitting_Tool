const form = document.getElementById('processForm');
const submitBtn = document.getElementById('submitBtn');
const progressCard = document.getElementById('progressCard');
const resultsCard = document.getElementById('resultsCard');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const resultsContent = document.getElementById('resultsContent');

let statusInterval;

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const url = document.getElementById('url').value.trim();
    
    if (!url) {
        alert('Please provide a video URL');
        return;
    }
    
    submitBtn.disabled = true;
    progressCard.style.display = 'block';
    resultsCard.style.display = 'none';
    
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ url }),
        });
        
        const data = await response.json();
        
        if (data.status === 'started') {
            startStatusPolling();
        }
    } catch (error) {
        alert('Error starting process: ' + error.message);
        submitBtn.disabled = false;
        progressCard.style.display = 'none';
    }
});

function startStatusPolling() {
    statusInterval = setInterval(async () => {
        try {
            const response = await fetch('/status');
            const status = await response.json();
            
            updateProgress(status);
            
            if (status.status === 'completed') {
                clearInterval(statusInterval);
                showResults(status);
                submitBtn.disabled = false;
            } else if (status.status === 'error') {
                clearInterval(statusInterval);
                showError(status.message);
                submitBtn.disabled = false;
            }
        } catch (error) {
            console.error('Error polling status:', error);
        }
    }, 1000);
}

function updateProgress(status) {
    progressFill.style.width = status.progress + '%';
    progressText.textContent = status.message;
}

async function showResults(status) {
    progressCard.style.display = 'none';
    resultsCard.style.display = 'block';
    
    let html = '';
    
    if (status.clips && status.clips.length > 0) {
        html += '<div class="result-item"><h4>🎬 Generated Clips</h4><div class="clips-grid">';
        status.clips.forEach((clip, index) => {
            const clipName = clip.split('\\').pop().split('/').pop();
            html += `
                <div class="clip-item" style="display: block;">
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                        <div class="clip-icon">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                                <polygon points="5 3 19 12 5 21 5 3"/>
                            </svg>
                        </div>
                        <div class="clip-info">
                            <div class="clip-name">Clip ${index + 1}</div>
                            <div class="clip-path">${clipName}</div>
                        </div>
                    </div>
                    <video controls style="width: 100%; border-radius: 0.5rem; margin-bottom: 0.5rem;">
                        <source src="/view/${encodeURIComponent(clip)}" type="video/mp4">
                    </video>
                    <div style="margin-top: 8px;">
                        <a href="/download/${encodeURIComponent(clip)}" download>⬇ Download</a>
                    </div>
                </div>
            `;
        });
        html += '</div></div>';
    }
    
    if (status.transcript) {
        const transcriptData = await fetch(`/view/${encodeURIComponent(status.transcript)}`).then(r => r.json()).catch(() => null);
        html += `
            <div class="result-item">
                <h4>📝 Transcript</h4>
                <div class="transcript-viewer" style="max-height: 300px; overflow-y: auto; background: #0f172a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">`;
        if (transcriptData && transcriptData.segments) {
            transcriptData.segments.forEach(seg => {
                html += `<p style="margin-bottom: 0.5rem;"><strong>[${seg.start.toFixed(1)}s - ${seg.end.toFixed(1)}s]</strong> ${seg.text}</p>`;
            });
        }
        html += `</div>
                <p>
                    <a href="/download/txt/${encodeURIComponent(status.transcript)}" download>⬇ Download TXT</a>
                    <a href="/download/pdf/${encodeURIComponent(status.transcript)}" download style="margin-left: 10px;">⬇ Download PDF</a>
                </p>
            </div>
        `;
    }
    
    if (status.segments) {
        const segmentsData = await fetch(`/view/${encodeURIComponent(status.segments)}`).then(r => r.json()).catch(() => null);
        html += `
            <div class="result-item">
                <h4>📊 Segments</h4>
                <div class="segments-viewer" style="max-height: 200px; overflow-y: auto; background: #0f172a; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0;">`;
        if (segmentsData && Array.isArray(segmentsData)) {
            segmentsData.forEach(seg => {
                html += `<p style="margin-bottom: 0.75rem;"><strong>Segment ${seg.segment_id}:</strong> ${seg.text.substring(0, 100)}...</p>`;
            });
        }
        html += `</div>
                <p>
                    <a href="/download/txt/${encodeURIComponent(status.segments)}" download>⬇ Download TXT</a>
                    <a href="/download/pdf/${encodeURIComponent(status.segments)}" download style="margin-left: 10px;">⬇ Download PDF</a>
                </p>
            </div>
        `;
    }
    
    resultsContent.innerHTML = html;
}

function showError(message) {
    progressCard.style.display = 'none';
    resultsCard.style.display = 'block';
    resultsCard.querySelector('h3').innerHTML = `
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        Error Occurred
    `;
    resultsCard.querySelector('h3').style.color = '#ef4444';
    resultsContent.innerHTML = `
        <div class="result-item" style="border-left-color: #ef4444;">
            <p style="color: #ef4444;">${message}</p>
        </div>
    `;
}
