document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-btn');
    
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
                indicator.className = 'message ai';
                indicator.innerHTML = `
                    <div class="typing-indicator" style="display: flex;">
                        <span></span><span></span><span></span>
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

        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = text;
        msgDiv.appendChild(bubble);

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
                // Optionally add a tooltip with the excerpt
                badge.title = src.excerpt;
                sourcesContainer.appendChild(badge);
            });
            msgDiv.appendChild(sourcesContainer);
        }

        chatContainer.appendChild(msgDiv);
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
            const res = await fetch('/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: text,
                    session_id: sessionId
                })
            });
            
            if (!res.ok) {
                const errorData = await res.json();
                appendMessage(errorData.detail || 'An error occurred.', 'ai');
            } else {
                const data = await res.json();
                appendMessage(data.answer, 'ai', data.sources);
            }
        } catch (err) {
            appendMessage('Network error. Is the server running?', 'ai');
        } finally {
            setTyping(false);
        }
    });

    clearBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear the chat history?')) return;
        
        try {
            await fetch(`/history/${sessionId}`, { method: 'DELETE' });
        } catch (e) {
            console.error(e);
        }
        
        chatContainer.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to Regulify RAG</h2>
                <p>History cleared. Ask a new question.</p>
            </div>
        `;
    });
});
