/**
 * MyLawLLM - Legal Assistant Frontend
 * Production-grade JavaScript module
 */

(function() {
  'use strict';

  // ============================================
  // CONFIGURATION
  // ============================================
  const CONFIG = {
    API_ENDPOINT: '/ask',
    DEBOUNCE_DELAY: 150,
    MAX_INPUT_HEIGHT: 150,
    ANIMATION_DURATION: 400
  };

  // ============================================
  // STATE MANAGEMENT
  // ============================================
  const state = {
    history: [],
    isProcessing: false,
    elements: {}
  };

  // ============================================
  // DOM CACHE
  // ============================================
  function cacheElements() {
    state.elements = {
      messages: document.getElementById('messages'),
      input: document.getElementById('input'),
      sendBtn: document.getElementById('send'),
      sourcesList: document.getElementById('sources-list'),
      sourcesPanel: document.querySelector('.sources-panel')
    };
  }

  // ============================================
  // UTILITY FUNCTIONS
  // ============================================
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function formatAnswer(text) {
    let formatted = text;

    // Split into sections based on headers
    const plainMatch = formatted.match(/\*\*Plain-English Answer\*\*[:\s]*([\s\S]*?)(?=\*\*Legal Basis\*\*|$)/i) 
                    || formatted.match(/Plain-English Answer[:\s]*([\s\S]*?)(?=Legal Basis|$)/i);
    const legalMatch = formatted.match(/\*\*Legal Basis\*\*[:\s]*([\s\S]*$)/i) 
                    || formatted.match(/Legal Basis[:\s]*([\s\S]*$)/i);

    if (plainMatch || legalMatch) {
      let html = '<div class="answer">';

      if (plainMatch && plainMatch[1]) {
        let content = plainMatch[1].trim();
        // Clean up numbered list items but keep numbers
        content = content.replace(/^\d+\.\s*/gm, '');
        
        html += `<div class="answer-section answer-section--plain">
          <div class="answer-section__title">Plain English Answer</div>
          <div class="answer-section__content">${formatListItems(highlightCitations(content))}</div>
        </div>`;
      }

      if (legalMatch && legalMatch[1]) {
        let content = legalMatch[1].trim();
        // Clean up numbered list items but keep numbers  
        content = content.replace(/^\d+\.\s*/gm, '');
        
        html += `<div class="answer-section answer-section--legal">
          <div class="answer-section__title">Legal Basis</div>
          <div class="answer-section__content">${formatListItems(highlightCitations(content))}</div>
        </div>`;
      }

      html += '</div>';
      return html;
    }

    // Fallback: just format the text normally
    return `<div class="answer">
      <div class="answer-section answer-section--plain">
        <div class="answer-section__title">Answer</div>
        <div class="answer-section__content">${formatListItems(highlightCitations(formatted))}</div>
      </div>
    </div>`;
  }

  function formatListItems(text) {
    // Split by numbered items (1., 2., etc) or bullet points
    const lines = text.split('\n');
    let html = '';
    let inList = false;

    lines.forEach(line => {
      const trimmed = line.trim();
      if (!trimmed) {
        if (inList) { html += '</ul>'; inList = false; }
        return;
      }

      // Handle numbered items or bullet points
      if (/^\d+[\.\)]\s*/.test(trimmed) || /^[-•●]\s*/.test(trimmed)) {
        if (!inList) { html += '<ul class="answer-list">'; inList = true; }
        const content = trimmed.replace(/^(\d+[\.\)]|[-•●])\s*/, '');
        html += `<li>${content}</li>`;
      } else {
        if (inList) { html += '</ul>'; inList = false; }
        html += `<p>${trimmed}</p>`;
      }
    });

    if (inList) html += '</ul>';
    return html || `<p>${text}</p>`;
  }

  function formatParagraphs(text) {
    // Split into paragraphs
    const paragraphs = text.split(/\n\n+/).filter(p => p.trim());
    
    if (paragraphs.length === 0) {
      return `<p>${text}</p>`;
    }

    return paragraphs.map(p => `<p>${p.trim()}</p>`).join('');
  }

  function highlightCitations(text) {
    let result = text;

    // Remove markdown asterisks first
    result = result.replace(/\*\*/g, '');

    // Replace legal document names with sections - handles Act, Ordinance, Code, Law
    // Pattern: "Name Type sections/section X"
    const legalTypes = ['Act', 'Ordinance', 'Code', 'Law'];
    
    legalTypes.forEach(type => {
      const pattern = new RegExp(`([A-Z][a-zA-Z]+(?:\\s+[A-Z]?[a-zA-Z]+)*\\s+${type}[\\w\\s]*)\\b(,?\\s*(?:sections|section|s\\.)\\s*\\d+(?:[\\s,]+(?:and\\s+)?\\d+)?)`, 'gi');
      result = result.replace(pattern, '<span class="citation">$1$2</span>');
    });

    // Replace standalone section references with pink highlight
    result = result.replace(/\b(sections?|s\.)\s*(\d+(?:[\s,]+(?:and\s+)?\d+)?)\b/gi,
      '<span class="citation citation--section">$1 $2</span>');

    // Highlight specific legal doctrines
    result = result.replace(/\b(Roman-Dutch Law)\b/gi,
      '<span class="citation">$1</span>');
    result = result.replace(/\b(Constitution of Sri Lanka)\b/gi,
      '<span class="citation">$1</span>');
    result = result.replace(/\b(Tort Law)\b/gi,
      '<span class="citation">$1</span>');
    result = result.replace(/\b(Law of Delict)\b/gi,
      '<span class="citation">$1</span>');

    return result;
  }

  function scrollToBottom() {
    const { messages } = state.elements;
    if (messages) {
      messages.scrollTo({
        top: messages.scrollHeight,
        behavior: 'smooth'
      });
    }
  }

  // ============================================
  // DOM MANIPULATION
  // ============================================
  function createMessageElement(role, content, isHtml = false) {
    const wrapper = document.createElement('div');
    wrapper.className = `message ${role === 'user' ? 'user' : 'bot'}`;

    const label = document.createElement('div');
    label.className = 'label';
    label.textContent = role === 'user' ? 'You' : 'MyLawLLM';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    if (isHtml) {
      bubble.innerHTML = content;
    } else {
      bubble.textContent = content;
    }

    wrapper.appendChild(label);
    wrapper.appendChild(bubble);

    return wrapper;
  }

  function createThinkingIndicator() {
    return `
      <div class="thinking">
        <div class="thinking__dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <span>Consulting legal database...</span>
      </div>
    `;
  }

  function addMessage(role, content, isHtml = false) {
    const { messages } = state.elements;
    if (!messages) return null;

    const messageEl = createMessageElement(role, content, isHtml);
    messages.appendChild(messageEl);
    scrollToBottom();

    return messageEl.querySelector('.bubble');
  }

  function clearSources() {
    const { sourcesList } = state.elements;
    if (!sourcesList) return;

    sourcesList.innerHTML = `
      <div class="sources-empty">
        <div class="sources-empty__icon">📜</div>
        <div class="sources-empty__text">Your legal sources will appear here after your first question.</div>
      </div>
    `;
  }

  function renderSources(sources) {
    const { sourcesList } = state.elements;
    if (!sourcesList) return;

    if (!sources || sources.length === 0) {
      sourcesList.innerHTML = `
        <div class="sources-empty">
          <div class="sources-empty__icon">🔍</div>
          <div class="sources-empty__text">No relevant sources found</div>
        </div>
      `;
      return;
    }

    sourcesList.innerHTML = '';
    sources.forEach(source => {
      const card = document.createElement('div');
      card.className = 'source-card';
      card.setAttribute('role', 'button');
      card.setAttribute('tabindex', '0');

      const nameDiv = document.createElement('div');
      nameDiv.className = 'source-card__name';
      nameDiv.textContent = formatSourceName(source.source);

      const excerptDiv = document.createElement('div');
      excerptDiv.className = 'source-card__excerpt';
      excerptDiv.textContent = source.excerpt;

      card.appendChild(nameDiv);
      card.appendChild(excerptDiv);
      sourcesList.appendChild(card);
    });
  }

  function formatSourceName(source) {
    return source
      .replace(/_/g, ' ')
      .replace(/\.pdf$/i, '')
      .replace(/_/g, ' ')
      .trim();
  }

  // ============================================
  // INPUT HANDLING
  // ============================================
  function autoResizeInput() {
    const { input } = state.elements;
    if (!input) return;

    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, CONFIG.MAX_INPUT_HEIGHT) + 'px';
  }

  function setInputValue(value) {
    const { input } = state.elements;
    if (!input) return;
    input.value = value;
    autoResizeInput();
  }

  function getInputValue() {
    const { input } = state.elements;
    return input ? input.value.trim() : '';
  }

  function setLoadingState(isLoading) {
    const { sendBtn, input } = state.elements;
    state.isProcessing = isLoading;

    if (sendBtn) {
      sendBtn.disabled = isLoading;
      sendBtn.setAttribute('aria-disabled', isLoading);
    }

    if (input) {
      input.disabled = isLoading;
    }
  }

  // ============================================
  // API COMMUNICATION
  // ============================================
  async function fetchAnswer(question) {
    const payload = {
      question: question,
      history: state.history
    };

    const response = await fetch(CONFIG.API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  // ============================================
  // EVENT HANDLERS
  // ============================================
  async function handleSendQuestion() {
    if (state.isProcessing) return;

    const question = getInputValue();
    if (!question) return;

    setInputValue('');
    setLoadingState(true);
    clearSources();

    addMessage('user', escapeHtml(question));

    const thinkingBubble = addMessage('bot', createThinkingIndicator(), true);

    try {
      const data = await fetchAnswer(question);

      if (thinkingBubble) {
        thinkingBubble.innerHTML = formatAnswer(data.answer);
      }

      renderSources(data.sources);

      state.history.push({ role: 'user', content: question });
      state.history.push({ role: 'assistant', content: data.answer });

    } catch (error) {
      console.error('Error fetching answer:', error);
      if (thinkingBubble) {
        thinkingBubble.innerHTML = '⚠️ Unable to connect to the server. Please ensure the backend is running.';
      }
    } finally {
      setLoadingState(false);
      scrollToBottom();
      state.elements.input?.focus();
    }
  }

  function handleInputKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSendQuestion();
    }
  }

  const handleInputInput = debounce(autoResizeInput, CONFIG.DEBOUNCE_DELAY);

  // ============================================
  // EVENT BINDING
  // ============================================
  function bindEvents() {
    const { sendBtn, input } = state.elements;

    if (sendBtn) {
      sendBtn.addEventListener('click', handleSendQuestion);
      sendBtn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleSendQuestion();
        }
      });
    }

    if (input) {
      input.addEventListener('keydown', handleInputKeydown);
      input.addEventListener('input', handleInputInput);
    }

    // Handle window resize for responsive adjustments
    window.addEventListener('resize', debounce(() => {
      autoResizeInput();
    }, 250));
  }

  // ============================================
  // INITIALIZATION
  // ============================================
  function init() {
    cacheElements();
    bindEvents();

    // Auto-focus input on load
    const { input } = state.elements;
    if (input) {
      input.focus();
    }

    // Log initialization
    console.log('⚖️ MyLawLLM initialized');
  }

  // ============================================
  // DOM READY
  // ============================================
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();