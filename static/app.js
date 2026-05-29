document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    const themeToggleBtn = document.getElementById('theme-toggle');
    const sessionList = document.getElementById('session-list');
    const newChatBtn = document.getElementById('new-chat-btn');

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
        } catch (err) {
            console.error(err);
            appendMessage('Network error. Is the server running?', 'ai');
        } finally {
            setTyping(false);
            if (typeof loadSessions === 'function') loadSessions();
        }
    });

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

            // Display most recent (bottom) first in sidebar
            [...data.sessions].reverse().forEach(sid => {
                const item = document.createElement('div');
                item.className = 'session-item';
                if (sid === sessionId) item.classList.add('active');
                item.textContent = sid;
                item.addEventListener('click', () => loadHistory(sid));
                sessionList.appendChild(item);
            });
        } catch (e) { console.error('Sessions fetch error:', e); }
    };

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

    newChatBtn.addEventListener('click', () => {
        sessionId = 'session_' + Math.random().toString(36).substring(2, 10);
        sessionStorage.setItem('sessionId', sessionId);
        loadHistory(sessionId);
    });

    // Initialize App
    loadHistory(sessionId);
});
