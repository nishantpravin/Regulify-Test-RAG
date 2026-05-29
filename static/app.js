document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const sessionList = document.getElementById('session-list');
    const newChatBtn = document.getElementById('new-chat-btn');
    const documentList = document.getElementById('document-list');
    const addDocBtn = document.getElementById('add-doc-btn');
    const docUploadInput = document.getElementById('doc-upload-input');

    let currentDocId = localStorage.getItem('currentDocId') || null;

    // Theme Management
    let currentTheme = localStorage.getItem('theme') || 'dark';
    if (currentTheme === 'light') document.body.classList.add('light-theme');
    themeToggleBtn.addEventListener('click', () => {
        currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
        localStorage.setItem('theme', currentTheme);
        if (currentTheme === 'light') document.body.classList.add('light-theme');
        else document.body.classList.remove('light-theme');
    });

    // Generate a permanent session ID for this browser tab
    let sessionId = sessionStorage.getItem('sessionId');
    if (!sessionId) {
        sessionId = 'session_' + Math.random().toString(36).substring(2, 10);
        sessionStorage.setItem('sessionId', sessionId);
    }

    const setTyping = (isTyping) => {
        let indicator = document.getElementById('typing-indicator');
        if (isTyping) {
            if (!indicator) {
                indicator = document.createElement('div');
                indicator.id = 'typing-indicator';
                indicator.className = 'message-row ai';
                indicator.innerHTML = `
                    <div class="ai-avatar">R</div>
                    <div class="bubble-wrap">
                        <div class="bubble">
                            <div class="typing-indicator" style="display: flex;">
                                <span></span><span></span><span></span>
                            </div>
                        </div>
                    </div>`;
                chatContainer.appendChild(indicator);
            }
            scrollToBottom();
            sendBtn.disabled = true;
            userInput.disabled = true;
        } else {
            if (indicator) indicator.remove();
            sendBtn.disabled = false;
            userInput.disabled = false;
            userInput.focus();
        }
    };

    const scrollToBottom = () => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    };

    const appendMessage = (text, sender, sources = []) => {
        // Remove welcome message if exists
        const welcome = document.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        const messageRow = document.createElement('div');
        messageRow.className = `message-row ${sender}`;

        // Add AI Avatar if sender is AI
        if (sender === 'ai') {
            const avatar = document.createElement('div');
            avatar.className = 'ai-avatar';
            avatar.textContent = 'R'; // R for Regulify
            messageRow.appendChild(avatar);
        }

        const bubbleWrap = document.createElement('div');
        bubbleWrap.className = 'bubble-wrap';

        const bubble = document.createElement('div');
        bubble.className = 'bubble';

        const time = document.createElement('div');
        time.className = 'msg-time';
        const now = new Date();
        time.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        if (text && typeof marked !== 'undefined') {
            const mathTokens = [];
            let processedText = text;
            processedText = processedText.replace(/(\$\$|\\\[)([\s\S]*?)(\$\$|\\\])/g, (match) => {
                mathTokens.push(match); return `%%MATH_TOKEN_${mathTokens.length - 1}%%`;
            });
            processedText = processedText.replace(/(\\\()([\s\S]*?)(\\\))/g, (match) => {
                mathTokens.push(match); return `%%MATH_TOKEN_${mathTokens.length - 1}%%`;
            });

            const rawHtml = marked.parse(processedText);
            let cleanHtml = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(rawHtml) : rawHtml;
            mathTokens.forEach((token, index) => {
                cleanHtml = cleanHtml.replace(`%%MATH_TOKEN_${index}%%`, () => token);
            });

            bubble.innerHTML = cleanHtml;

            if (typeof renderMathInElement !== 'undefined') {
                renderMathInElement(bubble, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '\\[', right: '\\]', display: true },
                        { left: '\\(', right: '\\)', display: false }
                    ],
                    throwOnError: false
                });
            }
        } else if (text) {
            bubble.textContent = text;
        }

        bubbleWrap.appendChild(bubble);

        if (sources && sources.length > 0) {
            const sourcesContainer = document.createElement('div');
            sourcesContainer.className = 'sources-container';
            const label = document.createElement('span');
            label.className = 'sources-label';
            label.textContent = 'Sources:';
            sourcesContainer.appendChild(label);

            sources.forEach(src => {
                const badge = document.createElement('span');
                badge.className = 'source-badge';
                badge.textContent = `Page ${src.page}`;
                badge.title = src.excerpt;
                sourcesContainer.appendChild(badge);
            });
            bubbleWrap.appendChild(sourcesContainer);
        }

        bubbleWrap.appendChild(time);
        messageRow.appendChild(bubbleWrap);
        chatContainer.appendChild(messageRow);
        scrollToBottom();

        // Add event listeners to source badges
        const badges = messageRow.querySelectorAll('.source-badge');
        badges.forEach(badge => {
            badge.style.cursor = 'pointer';
            badge.addEventListener('click', () => {
                const pageNum = parseInt(badge.textContent.replace('Page ', ''));
                if (currentDocId) openPDFViewer(currentDocId, pageNum);
            });
        });
    };

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = userInput.value.trim();
        if (!text) return;

        appendMessage(text, 'user');
        userInput.value = '';

        setTyping(true);

        try {
            const res = await fetch('/query_stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    session_id: sessionId
                })
            });

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}));
                appendMessage(errorData.detail || 'An error occurred.', 'ai');
                setTyping(false);
                return;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let accumulatedText = "";
            let dataBuffer = "";
            let currentAiBubble = null;

            setTyping(false); // Remove typing indicator, chunks have started

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                dataBuffer += decoder.decode(value, { stream: true });
                const lines = dataBuffer.split('\n\n');
                dataBuffer = lines.pop(); // Keep incomplete chunk

                for (let line of lines) {
                    if (line.startsWith('data: ')) {
                        let event;
                        try {
                            event = JSON.parse(line.substring(6));
                        } catch (e) {
                            continue;
                        }

                        if (event.type === 'sources') {
                            appendMessage("", "ai", event.sources);
                            const aiMessages = chatContainer.querySelectorAll('.message-row.ai');
                            currentAiBubble = aiMessages[aiMessages.length - 1].querySelector('.bubble');
                        } else if (event.type === 'chunk') {
                            accumulatedText += event.text;
                            if (currentAiBubble) {
                                let processedText = accumulatedText;
                                // Basic Markdown and Math parsing inline for speed
                                const mathTokens = [];
                                processedText = processedText.replace(/(\$\$|\\\[)([\s\S]*?)(\$\$|\\\])/g, (match) => {
                                    mathTokens.push(match); return `%%MATH_TOKEN_${mathTokens.length - 1}%%`;
                                });
                                processedText = processedText.replace(/(\\\()([\s\S]*?)(\\\))/g, (match) => {
                                    mathTokens.push(match); return `%%MATH_TOKEN_${mathTokens.length - 1}%%`;
                                });

                                const rawHtml = typeof marked !== 'undefined' ? marked.parse(processedText) : processedText;
                                let cleanHtml = typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(rawHtml) : rawHtml;

                                mathTokens.forEach((token, idx) => {
                                    cleanHtml = cleanHtml.replace(`%%MATH_TOKEN_${idx}%%`, () => token);
                                });

                                currentAiBubble.innerHTML = cleanHtml;

                                if (typeof renderMathInElement !== 'undefined') {
                                    renderMathInElement(currentAiBubble, {
                                        delimiters: [
                                            { left: '$$', right: '$$', display: true },
                                            { left: '\\[', right: '\\]', display: true },
                                            { left: '\\(', right: '\\)', display: false }
                                        ],
                                        throwOnError: false
                                    });
                                }
                                scrollToBottom();
                            }
                        } else if (event.type === 'error') {
                            if (!currentAiBubble) appendMessage(event.content, "ai");
                            else currentAiBubble.innerHTML += "<br><br><b>Error:</b> " + event.content;
                        }
                    }
                }
            }
        } finally {
            setTyping(false);
            loadSessions();
        }
    });

    const exportBtn = document.getElementById('export-btn');
    exportBtn.addEventListener('click', () => exportChatToPDF());

    const exportChatToPDF = () => {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        let y = 10;
        doc.setFontSize(20);
        doc.text("Regulify Chat Export", 10, y);
        y += 15;
        doc.setFontSize(12);

        const rows = chatContainer.querySelectorAll('.message-row');
        rows.forEach(row => {
            const role = row.classList.contains('user') ? 'User' : 'AI';
            const text = row.querySelector('.bubble').innerText;
            const splitText = doc.splitTextToSize(`${role}: ${text}`, 180);

            if (y + (splitText.length * 7) > 280) {
                doc.addPage();
                y = 10;
            }
            doc.text(splitText, 10, y);
            y += (splitText.length * 7) + 5;
        });

        doc.save(`Regulify_Chat_${sessionId}.pdf`);
    };

    clearBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear the chat history?')) return;

        try {
            await fetch(`/history/${sessionId}`, { method: 'DELETE' });
            loadHistory(sessionId); // Resets the UI
        } catch (e) {
            console.error(e);
        }
    });

    const loadSessions = async () => {
        try {
            const res = await fetch('/sessions');
            if (!res.ok) return;
            const data = await res.json();
            sessionList.innerHTML = '';

            [...data.sessions].reverse().forEach(session => {
                const item = document.createElement('div');
                item.className = 'session-item';
                if (session.id === sessionId) item.classList.add('active');
                item.textContent = session.title;
                item.addEventListener('click', () => loadHistory(session.id));
                sessionList.appendChild(item);
            });
        } catch (e) { console.error('Sessions fetch error:', e); }
    };

    const loadDocuments = async () => {
        try {
            const res = await fetch('/documents');
            const docs = await res.json();
            documentList.innerHTML = '';

            if (docs.length === 0) {
                documentList.innerHTML = '<div class="empty-state">No documents uploaded</div>';
            }

            docs.forEach(doc => {
                const item = document.createElement('div');
                item.className = 'doc-item';
                if (doc.id === currentDocId) item.classList.add('active');
                item.innerHTML = `
                    <span class="doc-icon">📄</span>
                    <span class="doc-name">${doc.filename}</span>
                `;
                item.addEventListener('click', () => selectDocument(doc.id));
                documentList.appendChild(item);
            });

            if (!currentDocId && docs.length > 0) {
                selectDocument(docs[0].id);
            }
        } catch (e) { console.error('Docs fetch error:', e); }
    };

    const selectDocument = (docId) => {
        currentDocId = docId;
        localStorage.setItem('currentDocId', docId);
        loadDocuments(); // Update active class
    };

    addDocBtn.addEventListener('click', () => docUploadInput.click());

    docUploadInput.addEventListener('change', async () => {
        const file = docUploadInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        setTyping(true); // Reuse typing logic for "Processing..."
        try {
            const res = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            if (res.ok) {
                const data = await res.json();
                selectDocument(data.doc_id);
                loadDocuments();
            } else {
                alert('Upload failed');
            }
        } catch (e) {
            console.error(e);
        } finally {
            setTyping(false);
            docUploadInput.value = '';
        }
    });

    const loadHistory = async (sid) => {
        sessionId = sid;
        sessionStorage.setItem('sessionId', sid);
        loadSessions(); // Trigger active highlight re-render

        try {
            const res = await fetch(`/history/${sid}`);
            if (!res.ok) return;
            const data = await res.json();

            chatContainer.innerHTML = '';
            if (!data.history || data.history.length === 0) {
                chatContainer.innerHTML = `
                    <div class="welcome-message">
                        <h2>Welcome to Regulify RAG</h2>
                        <p>Select a past chat from the left, or start typing to create a new one!</p>
                    </div>
                `;
                return;
            }

            data.history.forEach(msg => {
                appendMessage(msg.text, msg.sender);
            });
        } catch (e) { console.error('History fetch error:', e); }
    };

    newChatBtn.addEventListener('click', async () => {
        sessionId = 'session_' + Math.random().toString(36).substring(2, 10);
        sessionStorage.setItem('sessionId', sessionId);

        // Link new session to current document
        if (currentDocId) {
            await fetch('/session/doc', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, doc_id: currentDocId })
            });
        }
        loadHistory(sessionId);
    });

    // Voice Interaction (STT)
    const micBtn = document.createElement('button');
    micBtn.type = 'button';
    micBtn.id = 'mic-btn';
    micBtn.className = 'icon-btn mic-btn';
    micBtn.title = 'Voice to Text';
    micBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="23"></line>
            <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>`;
    chatForm.insertBefore(micBtn, sendBtn);

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;

        micBtn.addEventListener('click', () => {
            if (micBtn.classList.contains('recording')) {
                recognition.stop();
            } else {
                recognition.start();
                micBtn.classList.add('recording');
            }
        });

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            userInput.value += (userInput.value ? ' ' : '') + transcript;
            micBtn.classList.remove('recording');
            userInput.focus();
        };

        recognition.onerror = () => micBtn.classList.remove('recording');
        recognition.onend = () => micBtn.classList.remove('recording');
    } else {
        micBtn.style.display = 'none';
    }

    const viewerContainer = document.getElementById('pdf-viewer-container');
    const closeViewerBtn = document.getElementById('close-viewer');
    const pdfFilenameLabel = document.getElementById('pdf-filename');
    const renderArea = document.getElementById('pdf-render-area');

    let pdfDoc = null;

    closeViewerBtn.addEventListener('click', () => {
        viewerContainer.classList.add('hidden');
    });

    const openPDFViewer = async (docId, targetPage = 1) => {
        viewerContainer.classList.remove('hidden');
        renderArea.innerHTML = '<div class="loading">Loading PDF...</div>';

        try {
            const url = `/pdf/${docId}`;
            const loadingTask = pdfjsLib.getDocument(url);
            pdfDoc = await loadingTask.promise;
            renderArea.innerHTML = '';

            for (let i = 1; i <= pdfDoc.numPages; i++) {
                const canvas = document.createElement('canvas');
                canvas.id = `page-${i}`;
                canvas.className = 'pdf-page-canvas';
                renderArea.appendChild(canvas);

                const page = await pdfDoc.getPage(i);
                const viewport = page.getViewport({ scale: 1.5 });
                const context = canvas.getContext('2d');
                canvas.height = viewport.height;
                canvas.width = viewport.width;

                await page.render({ canvasContext: context, viewport: viewport }).promise;
            }

            // Scroll to target page
            const targetEl = document.getElementById(`page-${targetPage}`);
            if (targetEl) targetEl.scrollIntoView({ behavior: 'smooth' });

        } catch (e) {
            console.error('PDF error:', e);
            renderArea.innerHTML = '<div class="error">Failed to load PDF.</div>';
        }
    };

    // Initialize App
    loadDocuments();
    loadHistory(sessionId);
});
