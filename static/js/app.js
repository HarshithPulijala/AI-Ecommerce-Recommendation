/**
 * Main JavaScript for AI Product Recommender
 * Handles form submission, API calls, and UI updates
 */

// API Base URL
const API_BASE = '';

// DOM Elements
const form = document.getElementById('recommendationForm');
const userIdInput = document.getElementById('userId');
const topNSlider = document.getElementById('topN');
const topNValue = document.getElementById('topNValue');
const excludePurchasedCheckbox = document.getElementById('excludePurchased');
const submitBtn = document.getElementById('submitBtn');
const recommendationsContainer = document.getElementById('recommendationsContainer');
const statusContainer = document.getElementById('statusContainer');
const loadSampleBtn = document.getElementById('loadSampleBtn');

// State
let isLoading = false;

// ==================== EVENT LISTENERS ====================

// Update slider value display
topNSlider.addEventListener('input', (e) => {
    topNValue.textContent = e.target.value;
});

// Form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    await submitRecommendation();
});

// Load sample user
loadSampleBtn.addEventListener('click', async () => {
    await loadSampleUser();
});

// API Status link
document.getElementById('apiStatusLink').addEventListener('click', async (e) => {
    e.preventDefault();
    await showApiStatus();
});

// Stats link
document.getElementById('statsLink').addEventListener('click', async (e) => {
    e.preventDefault();
    await showStats();
});

// ==================== MAIN FUNCTIONS ====================

/**
 * Submit recommendation request
 */
async function submitRecommendation() {
    const userId = userIdInput.value.trim();
    const topN = parseInt(topNSlider.value);
    const excludePurchased = excludePurchasedCheckbox.checked;

    if (!userId) {
        showStatus('Please enter a user ID', 'error');
        return;
    }

    if (isLoading) return;

    isLoading = true;
    submitBtn.disabled = true;
    submitBtn.textContent = 'Loading...';
    showStatus('Fetching recommendations...', 'info');
    recommendationsContainer.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_id: userId,
                top_n: topN,
                exclude_purchased: excludePurchased
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.details || data.error || 'Failed to get recommendations');
        }

        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
        }

        if (data.recommendations.length === 0) {
            const statusType = data.valid_user === false ? 'warning' : 'warning';
            showStatus(data.message || 'No recommendations found for this user', statusType);
            showStatus(`Processed in ${data.processing_time_ms}ms`, 'success');
            recommendationsContainer.innerHTML = '<p class="no-results">No recommendations available</p>';
            return;
        }

        displayRecommendations(data.recommendations, data.processing_time_ms, data.message, data.valid_user);
        
        // Show appropriate status message
        const statusType = data.valid_user === false ? 'warning' : 'success';
        const statusMsg = data.message || `Successfully fetched ${data.recommendations.length} recommendations`;
        showStatus(`${statusMsg} (${data.processing_time_ms}ms)`, statusType);

    } catch (error) {
        console.error('Error:', error);
        showStatus(`Error: ${error.message}`, 'error');
    } finally {
        isLoading = false;
        submitBtn.disabled = false;
        submitBtn.textContent = 'Get Recommendations';
    }
}

/**
 * Load a sample user
 */
async function loadSampleUser() {
    try {
        showStatus('Loading sample users...', 'info');
        
        const response = await fetch(`${API_BASE}/api/users/sample?limit=1`);
        const data = await response.json();

        if (!data.success || !data.users || data.users.length === 0) {
            throw new Error('No sample users available');
        }

        userIdInput.value = data.users[0];
        showStatus(`Loaded sample user: ${data.users[0]}`, 'success');
    } catch (error) {
        console.error('Error:', error);
        showStatus(`Error loading sample user: ${error.message}`, 'error');
    }
}

/**
 * Display recommendations in grid
 */
function displayRecommendations(recommendations, processingTime, message = null, validUser = true) {
    recommendationsContainer.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'recommendations-header';
    
    // Add warning banner if invalid user
    let warningBanner = '';
    if (validUser === false) {
        warningBanner = `
            <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 12px; margin-bottom: 15px; color: #856404;">
                <strong>⚠️ Invalid User ID</strong><br>
                <span style="font-size: 0.9em;">The user ID you entered was not found. Showing popular products instead.</span>
            </div>
        `;
    }
    
    header.innerHTML = `
        ${warningBanner}
        <h2>📋 Top Recommendations</h2>
        <p>Processing time: ${processingTime}ms</p>
    `;
    recommendationsContainer.appendChild(header);

    const grid = document.createElement('div');
    grid.className = 'recommendations-grid';

    recommendations.forEach((rec, index) => {
        const card = createProductCard(rec, index + 1);
        grid.appendChild(card);
    });

    recommendationsContainer.appendChild(grid);
}

/**
 * Create a product recommendation card
 */
function createProductCard(product, rank) {
    const card = document.createElement('div');
    card.className = 'product-card';

    const ratingStars = createStarRating(product.predicted_rating);
    const priceDisplay = product.price > 0 ? `$${product.price.toFixed(2)}` : 'N/A';

    card.innerHTML = `
        <div class="product-rank">
            <span class="rank-badge">#${rank}</span>
        </div>
        
        <div class="product-header">
            <h3 class="product-title">${escapeHtml(product.title)}</h3>
        </div>

        <div class="product-meta">
            <span class="product-category">
                <strong>Category:</strong> ${escapeHtml(product.category)}
            </span>
            <span class="product-brand">
                <strong>Brand:</strong> ${escapeHtml(product.brand)}
            </span>
        </div>

        <div class="product-rating">
            <div class="stars">${ratingStars}</div>
            <span class="rating-value">${product.predicted_rating.toFixed(1)}/5.0</span>
        </div>

        <div class="product-price">
            <strong>Price:</strong> <span class="price-value">${priceDisplay}</span>
        </div>

        <div class="product-footer">
            <button class="btn-info" onclick="viewProductDetails('${product.product_id}')">
                View Details
            </button>
        </div>
    `;

    return card;
}

/**
 * Create star rating display
 */
function createStarRating(rating) {
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;
    const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);

    let stars = '★'.repeat(fullStars);
    if (hasHalfStar) stars += '½';
    stars += '☆'.repeat(emptyStars);

    return stars;
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    const statusDiv = document.createElement('div');
    statusDiv.className = `status-message status-${type}`;
    statusDiv.textContent = message;

    statusContainer.innerHTML = '';
    statusContainer.appendChild(statusDiv);

    // Auto-remove success/info messages after 5 seconds
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            if (statusContainer.contains(statusDiv)) {
                statusDiv.remove();
            }
        }, 5000);
    }
}

/**
 * View product details (placeholder)
 */
function viewProductDetails(productId) {
    showStatus(`Loading details for product ${productId}...`, 'info');
    
    fetch(`${API_BASE}/api/product/${productId}`)
        .then(r => r.json())
        .then(data => {
            if (data.success) {
                const product = data.product;
                const details = `
                    <strong>Product ID:</strong> ${product.product_id}<br>
                    <strong>Title:</strong> ${product.title}<br>
                    <strong>Category:</strong> ${product.category}<br>
                    <strong>Brand:</strong> ${product.brand}<br>
                    <strong>Price:</strong> $${product.price.toFixed(2)}<br>
                    <strong>Rating:</strong> ${product.mean_rating.toFixed(1)}/5.0 (${product.rating_count} ratings)
                `;
                alert(details);
            } else {
                throw new Error(data.error);
            }
        })
        .catch(error => {
            showStatus(`Error: ${error.message}`, 'error');
        });
}

/**
 * Show API Status
 */
async function showApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        const modal = document.getElementById('statusModal');
        const body = document.getElementById('statusModalBody');

        let statusHtml = `
            <div class="status-info">
                <p><strong>Status:</strong> ${data.status}</p>
                <p><strong>Models Loaded:</strong> ${data.models_loaded ? '✓ Yes' : '✗ No'}</p>
                <p><strong>Server Time:</strong> ${new Date(data.timestamp).toLocaleString()}</p>
                <p><strong>Initialized At:</strong> ${new Date(data.initialized_at).toLocaleString()}</p>
        `;

        if (data.models_error) {
            statusHtml += `<p class="error"><strong>Error:</strong> ${data.models_error}</p>`;
        }

        statusHtml += '</div>';
        body.innerHTML = statusHtml;
        openModal('statusModal');
    } catch (error) {
        showStatus(`Error fetching status: ${error.message}`, 'error');
    }
}

/**
 * Show System Stats
 */
async function showStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error);
        }

        const stats = data.stats;
        const statsHtml = `
            <div class="stats-grid">
                <div class="stat-item">
                    <span class="stat-label">Total Users</span>
                    <span class="stat-value">${stats.total_users.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Products</span>
                    <span class="stat-value">${stats.total_products.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Total Interactions</span>
                    <span class="stat-value">${stats.total_interactions.toLocaleString()}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">System Initialized</span>
                    <span class="stat-value">${new Date(stats.models_initialized_at).toLocaleString()}</span>
                </div>
            </div>
        `;

        const modal = document.getElementById('statsModal');
        const body = document.getElementById('statsModalBody');
        body.innerHTML = statsHtml;
        openModal('statsModal');
    } catch (error) {
        showStatus(`Error fetching stats: ${error.message}`, 'error');
    }
}

// ==================== MODAL FUNCTIONS ====================

function openModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'flex';
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.style.display = 'none';
}

// Close modal when clicking outside
window.addEventListener('click', (e) => {
    ['statusModal', 'statsModal'].forEach(modalId => {
        const modal = document.getElementById(modalId);
        if (e.target === modal) {
            closeModal(modalId);
        }
    });
});

// ==================== UTILITY FUNCTIONS ====================

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== INITIALIZATION ====================

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('AI Product Recommender loaded');
    
    // Check API health on startup
    fetch(`${API_BASE}/api/health`)
        .then(r => r.json())
        .then(data => {
            if (data.models_loaded) {
                showStatus('✓ System ready - All models loaded', 'success');
            } else {
                showStatus('⚠️ Models still loading...', 'warning');
            }
        })
        .catch(() => {
            showStatus('⚠️ Unable to connect to API', 'warning');
        });
});
