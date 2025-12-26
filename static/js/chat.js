/**
 * Modern Recipe Chatbot - Interactive JavaScript
 * Handles chat interactions, animations, recipe cards, and social sharing
 */

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    typingDelay: 1500,
    messageAnimationDelay: 300,
    defaultServings: 4,
};

// ============================================
// STATE MANAGEMENT
// ============================================
const state = {
    isTyping: false,
    currentRecipe: null,
    currentServings: CONFIG.defaultServings,
    chatHistory: [],
    theme: localStorage.getItem('theme') || 'light',
};

// ============================================
// DOM ELEMENTS
// ============================================
const elements = {
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    sendButton: document.getElementById('send-button'),
    themeToggle: document.getElementById('theme-toggle'),
    welcomeScreen: document.getElementById('welcome-screen'),
    typingIndicator: document.getElementById('typing-indicator'),
    modalOverlay: document.getElementById('modal-overlay'),
};

// ============================================
// THEME MANAGEMENT
// ============================================
function initTheme() {
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = elements.themeToggle?.querySelector('.theme-icon');
    if (icon) {
        icon.textContent = state.theme === 'light' ? '🌙' : '☀️';
    }
}

// ============================================
// CHAT FUNCTIONS
// ============================================
function getCurrentTime() {
    return new Date().toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

function hideWelcomeScreen() {
    if (elements.welcomeScreen) {
        elements.welcomeScreen.style.display = 'none';
    }
}

function showTypingIndicator() {
    if (elements.typingIndicator) {
        elements.typingIndicator.classList.remove('hidden');
        elements.typingIndicator.scrollIntoView({ behavior: 'smooth' });
    }
    state.isTyping = true;
}

function hideTypingIndicator() {
    if (elements.typingIndicator) {
        elements.typingIndicator.classList.add('hidden');
    }
    state.isTyping = false;
}

function addMessage(content, isUser = false, options = {}) {
    hideWelcomeScreen();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = isUser ? '👤' : '🍳';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Handle different content types
    if (typeof content === 'string') {
        contentDiv.innerHTML = content;
    } else if (content.type === 'recipe_cards') {
        contentDiv.innerHTML = createRecipeCardsHTML(content.recipes);
    } else if (content.type === 'recipe_detail') {
        contentDiv.innerHTML = createRecipeDetailHTML(content.recipe);
    }

    // Add timestamp
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = getCurrentTime();
    contentDiv.appendChild(timeDiv);

    // Add follow-up suggestions if provided
    if (options.suggestions && options.suggestions.length > 0) {
        const suggestionsDiv = createFollowUpSuggestions(options.suggestions);
        contentDiv.appendChild(suggestionsDiv);
    }

    // Add data source badge if provided
    if (options.dataSource) {
        const badge = createDataSourceBadge(options.dataSource);
        contentDiv.appendChild(badge);
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);

    // Insert before typing indicator
    if (elements.typingIndicator) {
        elements.chatMessages.insertBefore(messageDiv, elements.typingIndicator);
    } else {
        elements.chatMessages.appendChild(messageDiv);
    }

    // Scroll to bottom
    messageDiv.scrollIntoView({ behavior: 'smooth' });

    // Store in history
    state.chatHistory.push({ content, isUser, timestamp: new Date() });

    return messageDiv;
}

// ============================================
// RECIPE CARDS
// ============================================
function createRecipeCardsHTML(recipes) {
    if (!recipes || recipes.length === 0) {
        return '<p>No recipes found. Try a different search!</p>';
    }

    const cardsHTML = recipes.slice(0, 6).map((recipe, index) => `
    <div class="recipe-card" data-recipe-id="${recipe.id || index}" onclick="openRecipeModal(${JSON.stringify(recipe).replace(/"/g, '&quot;')})">
      <div class="recipe-card-front">
        <img src="${recipe.image || '/static/images/placeholder.jpg'}" alt="${recipe.name}" class="recipe-card-image" onerror="this.src='/static/images/placeholder.jpg'">
        <div class="recipe-card-overlay"></div>
        <div class="recipe-card-content">
          <h4 class="recipe-card-title">${recipe.name || 'Untitled Recipe'}</h4>
          <div class="recipe-card-meta">
            ${recipe.cuisine ? `<span class="recipe-tag cuisine">🍽️ ${recipe.cuisine}</span>` : ''}
            ${recipe.prep_time ? `<span class="recipe-tag time">⏱️ ${recipe.prep_time}</span>` : ''}
            ${recipe.diet ? `<span class="recipe-tag diet">🥗 ${recipe.diet}</span>` : ''}
          </div>
        </div>
      </div>
      <div class="recipe-card-back">
        <h4>🥘 Key Ingredients</h4>
        <ul class="ingredients-list">
          ${getIngredientsList(recipe.ingredients)}
        </ul>
      </div>
    </div>
  `).join('');

    return `<div class="recipe-cards-container">${cardsHTML}</div>`;
}

function getIngredientsList(ingredients) {
    if (!ingredients) return '<li>Ingredients not available</li>';

    const ingredientArray = typeof ingredients === 'string'
        ? ingredients.split(',').slice(0, 6)
        : ingredients.slice(0, 6);

    return ingredientArray.map(ing => `<li>${ing.trim()}</li>`).join('');
}

function createRecipeDetailHTML(recipe) {
    return `
    <div class="recipe-detail-inline">
      <h3>${recipe.name}</h3>
      <p>${recipe.description || ''}</p>
      <div class="share-buttons">
        <button class="share-btn whatsapp" onclick="shareToWhatsApp('${encodeURIComponent(recipe.name)}')" title="Share on WhatsApp">📱</button>
        <button class="share-btn twitter" onclick="shareToTwitter('${encodeURIComponent(recipe.name)}')" title="Share on Twitter">🐦</button>
        <button class="share-btn copy-link" onclick="copyRecipeLink('${recipe.id || ''}')" title="Copy Link">🔗</button>
      </div>
    </div>
  `;
}

// ============================================
// FOLLOW-UP SUGGESTIONS
// ============================================
function createFollowUpSuggestions(suggestions) {
    const container = document.createElement('div');
    container.className = 'follow-up-suggestions';

    suggestions.forEach(suggestion => {
        const chip = document.createElement('button');
        chip.className = 'suggestion-chip';
        chip.innerHTML = `<span class="icon">${suggestion.icon || '💡'}</span> ${suggestion.label}`;
        chip.onclick = () => handleSuggestionClick(suggestion);
        container.appendChild(chip);
    });

    return container;
}

function handleSuggestionClick(suggestion) {
    if (suggestion.action) {
        // Execute predefined action
        switch (suggestion.action) {
            case 'substitutions':
                sendMessage('What are some ingredient substitutions for this recipe?');
                break;
            case 'similar':
                sendMessage('Show me similar recipes');
                break;
            case 'favorite':
                addToFavorites(state.currentRecipe);
                break;
            case 'healthy':
                sendMessage('Why is this recipe healthy?');
                break;
            case 'scale':
                showScalingModal();
                break;
            default:
                sendMessage(suggestion.query || suggestion.label);
        }
    } else if (suggestion.query) {
        sendMessage(suggestion.query);
    }
}

function getDefaultSuggestions() {
    return [
        { label: 'Want substitutions?', icon: '🔄', action: 'substitutions' },
        { label: 'Similar recipes', icon: '👀', action: 'similar' },
        { label: 'Add to favorites', icon: '❤️', action: 'favorite' },
        { label: 'Health benefits', icon: '💚', action: 'healthy' },
    ];
}

// ============================================
// DATA SOURCE BADGE
// ============================================
function createDataSourceBadge(source) {
    const badge = document.createElement('div');
    badge.className = `data-source-badge ${source.type || 'local'}`;
    badge.innerHTML = `<span>${source.icon || '📊'}</span> ${source.label || 'Local Database'}`;
    return badge;
}

// ============================================
// RECIPE MODAL
// ============================================
function openRecipeModal(recipe) {
    state.currentRecipe = recipe;
    state.currentServings = CONFIG.defaultServings;

    const modal = elements.modalOverlay;
    if (!modal) return;

    modal.innerHTML = `
    <div class="modal-content">
      <div class="modal-header">
        <img src="${recipe.image || '/static/images/placeholder.jpg'}" alt="${recipe.name}" onerror="this.src='/static/images/placeholder.jpg'">
        <div class="modal-header-overlay"></div>
        <button class="modal-close" onclick="closeModal()">✕</button>
      </div>
      <div class="modal-body">
        <h2 class="modal-title">${recipe.name}</h2>
        
        <div class="recipe-card-meta">
          ${recipe.cuisine ? `<span class="recipe-tag cuisine">🍽️ ${recipe.cuisine}</span>` : ''}
          ${recipe.prep_time ? `<span class="recipe-tag time">⏱️ ${recipe.prep_time}</span>` : ''}
          ${recipe.diet ? `<span class="recipe-tag diet">🥗 ${recipe.diet}</span>` : ''}
          ${recipe.course ? `<span class="recipe-tag">📍 ${recipe.course}</span>` : ''}
        </div>
        
        ${recipe.description ? `<p style="margin: 1rem 0; color: var(--text-secondary);">${recipe.description}</p>` : ''}
        
        <div class="scaling-container">
          <span class="scaling-label">👥 Servings:</span>
          <div class="scaling-controls">
            <button class="scaling-btn" onclick="adjustServings(-1)">−</button>
            <span class="scaling-value" id="servings-value">${state.currentServings}</span>
            <button class="scaling-btn" onclick="adjustServings(1)">+</button>
          </div>
        </div>
        
        <div class="modal-section">
          <h3>🥘 Ingredients</h3>
          <ul class="ingredients-list" id="scaled-ingredients">
            ${formatIngredientsList(recipe.ingredients)}
          </ul>
        </div>
        
        <div class="modal-section">
          <h3>📝 Instructions</h3>
          <div class="instructions-steps">
            ${formatInstructions(recipe.instructions)}
          </div>
        </div>
        
        ${recipe.health_explanation ? `
        <div class="modal-section">
          <h3>💚 Health Benefits</h3>
          <div class="health-explanation">
            <div class="icon">🌿</div>
            <p>${recipe.health_explanation}</p>
          </div>
        </div>
        ` : ''}
        
        ${recipe.cooking_tips && recipe.cooking_tips.length > 0 ? `
        <div class="modal-section">
          <h3>💡 Cooking Tips</h3>
          <div class="cooking-tips">
            ${recipe.cooking_tips.map(tip => `<div class="cooking-tip">${tip}</div>`).join('')}
          </div>
        </div>
        ` : ''}
        
        <div class="modal-section">
          <h3>📤 Share Recipe</h3>
          <div class="share-buttons">
            <button class="share-btn whatsapp" onclick="shareToWhatsApp('${encodeURIComponent(recipe.name)}')" title="Share on WhatsApp">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347m-5.421 7.403h-.004a9.87 9.87 0 01-5.031-1.378l-.361-.214-3.741.982.998-3.648-.235-.374a9.86 9.86 0 01-1.51-5.26c.001-5.45 4.436-9.884 9.888-9.884 2.64 0 5.122 1.03 6.988 2.898a9.825 9.825 0 012.893 6.994c-.003 5.45-4.437 9.884-9.885 9.884m8.413-18.297A11.815 11.815 0 0012.05 0C5.495 0 .16 5.335.157 11.892c0 2.096.547 4.142 1.588 5.945L.057 24l6.305-1.654a11.882 11.882 0 005.683 1.448h.005c6.554 0 11.89-5.335 11.893-11.893a11.821 11.821 0 00-3.48-8.413z"/></svg>
            </button>
            <button class="share-btn twitter" onclick="shareToTwitter('${encodeURIComponent(recipe.name)}')" title="Share on Twitter/X">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg>
            </button>
            <button class="share-btn facebook" onclick="shareToFacebook('${encodeURIComponent(recipe.name)}')" title="Share on Facebook">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/></svg>
            </button>
            <button class="share-btn copy-link" onclick="copyRecipeLink('${recipe.id || recipe.name}')" title="Copy Link">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  `;

    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    const modal = elements.modalOverlay;
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function formatIngredientsList(ingredients) {
    if (!ingredients) return '<li>Ingredients not available</li>';

    const ingredientArray = typeof ingredients === 'string'
        ? ingredients.split(',')
        : ingredients;

    return ingredientArray.map(ing => `<li>${ing.trim()}</li>`).join('');
}

function formatInstructions(instructions) {
    if (!instructions) return '<p>Instructions not available</p>';

    const steps = typeof instructions === 'string'
        ? instructions.split('.').filter(s => s.trim())
        : instructions;

    return steps.map((step, index) => `
    <div class="instruction-step">
      <span class="step-number">${index + 1}</span>
      <span class="step-text">${step.trim()}</span>
    </div>
  `).join('');
}

// ============================================
// RECIPE SCALING
// ============================================
function adjustServings(delta) {
    const newServings = Math.max(1, Math.min(20, state.currentServings + delta));
    if (newServings === state.currentServings) return;

    const oldServings = state.currentServings;
    state.currentServings = newServings;

    // Update display
    const servingsValue = document.getElementById('servings-value');
    if (servingsValue) {
        servingsValue.textContent = newServings;
    }

    // Scale ingredients
    scaleIngredients(oldServings, newServings);
}

function scaleIngredients(oldServings, newServings) {
    const ingredientsList = document.getElementById('scaled-ingredients');
    if (!ingredientsList || !state.currentRecipe) return;

    const scaleFactor = newServings / oldServings;
    const items = ingredientsList.querySelectorAll('li');

    items.forEach(item => {
        const text = item.textContent;
        // Pattern to match numbers at the start (including fractions)
        const scaled = text.replace(/^(\d+(?:\/\d+)?(?:\s+\d+\/\d+)?)\s*/, (match, num) => {
            const value = parseQuantity(num);
            if (value !== null) {
                const newValue = value * scaleFactor;
                return formatQuantity(newValue) + ' ';
            }
            return match;
        });
        item.textContent = scaled;
    });
}

function parseQuantity(str) {
    str = str.trim();

    // Handle fractions like "1/2"
    if (str.includes('/')) {
        const parts = str.split(/\s+/);
        if (parts.length === 2) {
            // "1 1/2" format
            const whole = parseFloat(parts[0]);
            const [num, den] = parts[1].split('/').map(Number);
            return whole + (num / den);
        } else {
            // "1/2" format
            const [num, den] = str.split('/').map(Number);
            return num / den;
        }
    }

    return parseFloat(str) || null;
}

function formatQuantity(value) {
    if (Number.isInteger(value)) {
        return value.toString();
    }

    // Convert to fractions for common values
    const fractions = {
        0.25: '1/4',
        0.33: '1/3',
        0.5: '1/2',
        0.67: '2/3',
        0.75: '3/4',
    };

    const whole = Math.floor(value);
    const decimal = value - whole;

    for (const [dec, frac] of Object.entries(fractions)) {
        if (Math.abs(decimal - parseFloat(dec)) < 0.1) {
            return whole > 0 ? `${whole} ${frac}` : frac;
        }
    }

    return value.toFixed(1);
}

// ============================================
// SOCIAL SHARING
// ============================================
function shareToWhatsApp(recipeName) {
    const text = `🍳 Check out this amazing recipe: ${decodeURIComponent(recipeName)}!\n\nFound on Recipe Chatbot 🤖`;
    const url = `https://wa.me/?text=${encodeURIComponent(text)}`;
    window.open(url, '_blank');
}

function shareToTwitter(recipeName) {
    const text = `🍳 Just discovered this delicious recipe: ${decodeURIComponent(recipeName)}! #Cooking #Recipes #FoodLover`;
    const url = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`;
    window.open(url, '_blank');
}

function shareToFacebook(recipeName) {
    const text = `Check out this amazing recipe: ${decodeURIComponent(recipeName)}!`;
    const url = `https://www.facebook.com/sharer/sharer.php?quote=${encodeURIComponent(text)}`;
    window.open(url, '_blank');
}

function copyRecipeLink(recipeId) {
    const url = `${window.location.origin}/recipe/${recipeId}`;
    navigator.clipboard.writeText(url).then(() => {
        showToast('Link copied to clipboard! 📋');
    }).catch(() => {
        showToast('Failed to copy link');
    });
}

function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toast.style.cssText = `
    position: fixed;
    bottom: 100px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--text-primary);
    color: var(--bg-primary);
    padding: 12px 24px;
    border-radius: 8px;
    z-index: 2000;
    animation: toastSlide 0.3s ease;
  `;

    document.body.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}

// ============================================
// FAVORITES
// ============================================
async function addToFavorites(recipe) {
    if (!recipe) {
        showToast('No recipe selected');
        return;
    }

    try {
        const response = await fetch('/favorite', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ recipe_id: recipe.id || recipe.name }),
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.action === 'added' ? '❤️ Added to favorites!' : '💔 Removed from favorites');
        } else {
            showToast(data.message || 'Please log in to save favorites');
        }
    } catch (error) {
        showToast('Please log in to save favorites');
    }
}

// ============================================
// MESSAGE SENDING
// ============================================
async function sendMessage(message = null) {
    const inputValue = message || elements.chatInput?.value.trim();

    if (!inputValue) return;

    // Clear input
    if (elements.chatInput && !message) {
        elements.chatInput.value = '';
        autoResizeInput();
    }

    // Add user message
    addMessage(inputValue, true);

    // Show typing indicator
    showTypingIndicator();

    // Disable send button
    if (elements.sendButton) {
        elements.sendButton.disabled = true;
    }

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: inputValue }),
        });

        const data = await response.json();

        // Simulate typing delay for better UX
        await new Promise(resolve => setTimeout(resolve, CONFIG.typingDelay));

        hideTypingIndicator();

        // Parse response and add with suggestions
        const suggestions = data.suggestions || getDefaultSuggestions();
        const dataSource = data.source ? {
            type: data.source.includes('API') ? 'api' : 'local',
            label: data.source,
            icon: data.source.includes('API') ? '🌐' : '💾'
        } : null;

        addMessage(data.response, false, { suggestions, dataSource });

        // Store current recipe if available
        if (data.recipe) {
            state.currentRecipe = data.recipe;
        }

        // Save chat to server
        saveChatToServer(inputValue);

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        addMessage('Sorry, something went wrong. Please try again! 😅', false);
    } finally {
        if (elements.sendButton) {
            elements.sendButton.disabled = false;
        }
        elements.chatInput?.focus();
    }
}

async function saveChatToServer(message) {
    try {
        await fetch('/save_chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });
    } catch (error) {
        // Silent fail for chat saving
    }
}

// ============================================
// INPUT HANDLING
// ============================================
function autoResizeInput() {
    if (elements.chatInput) {
        elements.chatInput.style.height = 'auto';
        elements.chatInput.style.height = Math.min(elements.chatInput.scrollHeight, 150) + 'px';
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function handleQuickAction(action) {
    const queries = {
        'breakfast': 'Show me some breakfast recipes',
        'lunch': 'What can I make for lunch?',
        'dinner': 'Suggest some dinner recipes',
        'vegetarian': 'Show me vegetarian recipes',
        'quick': 'Quick recipes under 30 minutes',
        'indian': 'Show me Indian cuisine recipes',
        'healthy': 'Healthy recipe suggestions',
    };

    sendMessage(queries[action] || action);
}

function handleWelcomeSuggestion(query) {
    sendMessage(query);
}

// ============================================
// INITIALIZATION
// ============================================
function init() {
    // Initialize theme
    initTheme();

    // Add event listeners
    if (elements.themeToggle) {
        elements.themeToggle.addEventListener('click', toggleTheme);
    }

    if (elements.sendButton) {
        elements.sendButton.addEventListener('click', () => sendMessage());
    }

    if (elements.chatInput) {
        elements.chatInput.addEventListener('keypress', handleKeyPress);
        elements.chatInput.addEventListener('input', autoResizeInput);
    }

    // Close modal on overlay click
    if (elements.modalOverlay) {
        elements.modalOverlay.addEventListener('click', (e) => {
            if (e.target === elements.modalOverlay) {
                closeModal();
            }
        });
    }

    // Close modal on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
        }
    });

    // Focus input on load
    elements.chatInput?.focus();

    console.log('🍳 Recipe Chatbot initialized!');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Add toast animation
const style = document.createElement('style');
style.textContent = `
  @keyframes toastSlide {
    from {
      opacity: 0;
      transform: translateX(-50%) translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateX(-50%) translateY(0);
    }
  }
`;
document.head.appendChild(style);
