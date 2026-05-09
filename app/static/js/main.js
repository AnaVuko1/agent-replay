/* Agent Replay - Main JavaScript */

// Theme Management
function toggleTheme() {
    const currentTheme = localStorage.getItem('theme') || 'dark';
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update button text
    const themeBtn = document.querySelector('[onclick="toggleTheme()"]');
    if (themeBtn) {
        themeBtn.innerHTML = newTheme === 'dark' 
            ? '<i class="fas fa-moon"></i> Dark' 
            : '<i class="fas fa-sun"></i> Light';
    }
}

// Initialize theme on page load
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
}

// Modal Management
function showModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.add('show');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.classList.remove('show');
        document.body.style.overflow = 'auto';
    }
}

// Close modal when clicking outside
document.addEventListener('click', function(event) {
    if (event.target.classList.contains('modal')) {
        closeModal(event.target.id);
    }
});

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => closeModal(modal.id));
    }
});

// Dropdown Management
document.addEventListener('click', function(event) {
    // Close all dropdowns when clicking outside
    if (!event.target.closest('.dropdown')) {
        document.querySelectorAll('.dropdown.show').forEach(dropdown => {
            dropdown.classList.remove('show');
        });
    }
    
    // Toggle dropdown when clicking toggle button
    if (event.target.classList.contains('dropdown-toggle') || 
        event.target.closest('.dropdown-toggle')) {
        const dropdown = event.target.closest('.dropdown');
        dropdown.classList.toggle('show');
        event.stopPropagation();
    }
});

// API Client
const API = {
    baseUrl: window.location.origin,
    
    async request(endpoint, options = {}) {
        const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;
        
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };
        
        try {
            const response = await fetch(url, { ...defaultOptions, ...options });
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({
                    detail: `HTTP ${response.status}: ${response.statusText}`
                }));
                throw new Error(error.detail || response.statusText);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${endpoint}`, error);
            throw error;
        }
    },
    
    // Sessions
    async getSessions(params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/v1/sessions?${query}`);
    },
    
    async getSession(sessionId) {
        return this.request(`/api/v1/sessions/${sessionId}`);
    },
    
    async createSession(sessionData) {
        return this.request('/api/v1/sessions', {
            method: 'POST',
            body: JSON.stringify(sessionData)
        });
    },
    
    async deleteSession(sessionId) {
        return this.request(`/api/v1/sessions/${sessionId}`, {
            method: 'DELETE'
        });
    },
    
    // Steps
    async getSessionSteps(sessionId, params = {}) {
        const query = new URLSearchParams(params).toString();
        return this.request(`/api/v1/steps/session/${sessionId}?${query}`);
    },
    
    async getStep(stepId) {
        return this.request(`/api/v1/steps/${stepId}`);
    },
    
    async getStateSnapshot(sessionId, stepNumber) {
        return this.request(`/api/v1/steps/session/${sessionId}/snapshot?step=${stepNumber}`);
    },
    
    // Replays
    async getReplays(sessionId) {
        return this.request(`/api/v1/replays/session/${sessionId}/replays`);
    },
    
    async createReplay(sessionId, replayData) {
        return this.request(`/api/v1/replays/session/${sessionId}/replay`, {
            method: 'POST',
            body: JSON.stringify(replayData)
        });
    },
    
    async executeReplay(replayId) {
        return this.request(`/api/v1/replays/${replayId}/execute`, {
            method: 'POST'
        });
    },
    
    // Dashboard
    async getDashboardStats() {
        return this.request('/api/v1/dashboard/stats');
    },
    
    async getSystemHealth() {
        return this.request('/api/v1/dashboard/health');
    },
    
    // Compare
    async compareSessions(session1Id, session2Id) {
        return this.request(`/api/v1/compare/sessions?session1_id=${session1Id}&session2_id=${session2Id}`);
    },
    
    async compareReplays(replay1Id, replay2Id) {
        return this.request(`/api/v1/compare/compare_two_replays?replay1_id=${replay1Id}&replay2_id=${replay2Id}`);
    }
};

// Notification System
const Notifications = {
    container: null,
    
    init() {
        this.container = document.createElement('div');
        this.container.className = 'notifications';
        document.body.appendChild(this.container);
    },
    
    show(message, type = 'info', duration = 5000) {
        if (!this.container) this.init();
        
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas fa-${this.getIcon(type)}"></i>
            </div>
            <div class="notification-content">${message}</div>
            <button class="notification-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        this.container.appendChild(notification);
        
        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, duration);
        }
        
        return notification;
    },
    
    getIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    },
    
    success(message, duration = 3000) {
        return this.show(message, 'success', duration);
    },
    
    error(message, duration = 5000) {
        return this.show(message, 'error', duration);
    },
    
    warning(message, duration = 4000) {
        return this.show(message, 'warning', duration);
    },
    
    info(message, duration = 3000) {
        return this.show(message, 'info', duration);
    }
};

// Add notification styles
const notificationStyles = `
.notifications {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: 400px;
}

.notification {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    box-shadow: var(--shadow-lg);
    animation: notificationSlideIn 0.3s ease;
    max-width: 100%;
}

.notification-success {
    border-left: 4px solid var(--success-color);
}

.notification-error {
    border-left: 4px solid var(--danger-color);
}

.notification-warning {
    border-left: 4px solid var(--warning-color);
}

.notification-info {
    border-left: 4px solid var(--primary-color);
}

.notification-icon {
    color: var(--text-muted);
    font-size: 1.25rem;
    flex-shrink: 0;
}

.notification-success .notification-icon {
    color: var(--success-color);
}

.notification-error .notification-icon {
    color: var(--danger-color);
}

.notification-warning .notification-icon {
    color: var(--warning-color);
}

.notification-info .notification-icon {
    color: var(--primary-color);
}

.notification-content {
    flex: 1;
    color: var(--text-primary);
    font-size: 0.875rem;
    line-height: 1.5;
}

.notification-close {
    background: none;
    border: none;
    color: var(--text-muted);
    cursor: pointer;
    padding: 0;
    width: 1.5rem;
    height: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    flex-shrink: 0;
}

.notification-close:hover {
    background-color: var(--bg-tertiary);
    color: var(--text-primary);
}

@keyframes notificationSlideIn {
    from {
        opacity: 0;
        transform: translateX(100%);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}
`;

// Inject notification styles
const style = document.createElement('style');
style.textContent = notificationStyles;
document.head.appendChild(style);

// Format Helpers
const Format = {
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString();
    },
    
    formatDuration(ms) {
        if (!ms) return '0s';
        
        const seconds = Math.floor(ms / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    },
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    truncate(text, length = 100) {
        if (text.length <= length) return text;
        return text.substring(0, length) + '...';
    },
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Session Management Functions
function createNewSession() {
    showModal('createSessionModal');
}

async function submitNewSession() {
    const form = document.getElementById('createSessionForm');
    const formData = new FormData(form);
    
    const sessionData = {
        name: formData.get('sessionName'),
        agent_id: formData.get('agentId'),
        model: formData.get('model')
    };
    
    // Handle custom model
    if (sessionData.model === 'custom') {
        sessionData.model = formData.get('customModel');
    }
    
    // Parse metadata if provided
    const metadata = formData.get('metadata');
    if (metadata && metadata.trim()) {
        try {
            sessionData.metadata = JSON.parse(metadata);
        } catch (e) {
            Notifications.error('Invalid JSON in metadata');
            return;
        }
    }
    
    try {
        const session = await API.createSession(sessionData);
        Notifications.success('Session created successfully');
        closeModal('createSessionModal');
        
        // Redirect to session page
        setTimeout(() => {
            window.location.href = `/session/${session.id}`;
        }, 1000);
        
    } catch (error) {
        Notifications.error(`Failed to create session: ${error.message}`);
    }
}

function deleteSession() {
    if (!confirm('Are you sure you want to delete this session? This action cannot be undone.')) {
        return;
    }
    
    const sessionId = window.location.pathname.split('/').pop();
    
    API.deleteSession(sessionId)
        .then(() => {
            Notifications.success('Session deleted successfully');
            setTimeout(() => {
                window.location.href = '/dashboard';
            }, 1000);
        })
        .catch(error => {
            Notifications.error(`Failed to delete session: ${error.message}`);
        });
}

// Replay Management Functions
function startReplayFromSession() {
    showModal('createReplayModal');
}

async function submitNewReplay() {
    const sessionId = window.location.pathname.split('/').pop();
    const form = document.getElementById('createReplayForm');
    const formData = new FormData(form);
    
    const replayData = {
        name: formData.get('replayName'),
        start_step: parseInt(formData.get('replayStart')),
        end_step: parseInt(formData.get('replayEnd')),
        replay_config: {
            model: formData.get('replayModel'),
            temperature: parseFloat(formData.get('replayTemperature'))
        }
    };
    
    // Add system prompt if provided
    const systemPrompt = formData.get('replaySystemPrompt');
    if (systemPrompt && systemPrompt.trim()) {
        replayData.replay_config.system_prompt = systemPrompt.trim();
    }
    
    // Add constraints if provided
    const constraints = formData.get('replayConstraints');
    if (constraints && constraints.trim()) {
        replayData.replay_config.constraints = constraints
            .split('\n')
            .map(c => c.trim())
            .filter(c => c.length > 0);
    }
    
    try {
        const replay = await API.createReplay(sessionId, replayData);
        Notifications.success('Replay created successfully');
        closeModal('createReplayModal');
        
        // Refresh replay list
        loadReplays();
        
    } catch (error) {
        Notifications.error(`Failed to create replay: ${error.message}`);
    }
}

// Export Functions
function exportSession() {
    const sessionId = window.location.pathname.split('/').pop();
    window.open(`/api/v1/export/${sessionId}?format=json`, '_blank');
}

function copySessionLink() {
    const link = window.location.href;
    navigator.clipboard.writeText(link)
        .then(() => Notifications.success('Link copied to clipboard'))
        .catch(() => Notifications.error('Failed to copy link'));
}

// Navigation Functions
function goBack() {
    window.history.back();
}

function goToDashboard() {
    window.location.href = '/dashboard';
}

function goToSessions() {
    window.location.href = '/api/v1/sessions';
}

function goToReplays() {
    window.location.href = '/api/v1/replays';
}

// Utility Functions
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

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize theme
    initTheme();
    
    // Initialize notifications
    Notifications.init();
    
    // Set up dropdowns
    document.querySelectorAll('.dropdown').forEach(dropdown => {
        const toggle = dropdown.querySelector('.dropdown-toggle');
        if (toggle) {
            toggle.addEventListener('click', function(e) {
                e.stopPropagation();
                dropdown.classList.toggle('show');
            });
        }
    });
    
    // Close dropdowns when clicking outside
    document.addEventListener('click', function() {
        document.querySelectorAll('.dropdown.show').forEach(dropdown => {
            dropdown.classList.remove('show');
        });
    });
    
    // Handle form submissions
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
        });
    });
    
    // Add loading indicators to buttons
    document.addEventListener('click', function(e) {
        if (e.target.matches('.btn[type="submit"]') || 
            e.target.closest('.btn[type="submit"]')) {
            const btn = e.target.matches('.btn') ? e.target : e.target.closest('.btn');
            if (btn && !btn.classList.contains('loading')) {
                btn.classList.add('loading');
                btn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${btn.textContent}`;
                
                // Auto-remove loading after 30 seconds (safety)
                setTimeout(() => {
                    if (btn.classList.contains('loading')) {
                        btn.classList.remove('loading');
                        btn.innerHTML = btn.textContent.replace('Loading...', '').trim();
                    }
                }, 30000);
            }
        }
    });
    
    // Tooltip initialization
    document.querySelectorAll('[data-tooltip]').forEach(element => {
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = this.getAttribute('data-tooltip');
            
            const rect = this.getBoundingClientRect();
            tooltip.style.position = 'fixed';
            tooltip.style.top = (rect.top - 40) + 'px';
            tooltip.style.left = (rect.left + rect.width / 2) + 'px';
            tooltip.style.transform = 'translateX(-50%)';
            
            document.body.appendChild(tooltip);
            this._tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this._tooltip) {
                this._tooltip.remove();
                delete this._tooltip;
            }
        });
    });
});