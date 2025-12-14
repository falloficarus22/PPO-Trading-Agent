// Initialize Socket.IO connection
const socket = io();

// Chart instances
let portfolioChart = null;
let priceChart = null;

// State
let isTrading = false;
let portfolioData = [];
let priceData = [];
let actionData = [];

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadModels();
    initializeCharts();
    setupEventListeners();
    startPolling();
    
    // Socket.IO event handlers
    socket.on('connect', () => {
        addLog('Connected to server', 'info');
        socket.emit('request_status');
    });
    
    socket.on('status_update', (data) => {
        updateStatus(data);
    });
    
    socket.on('trading_update', (data) => {
        handleTradingUpdate(data);
    });
    
    socket.on('error', (data) => {
        addLog(`Error: ${data.message}`, 'error');
    });
});

function loadModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const select = document.getElementById('model-select');
            select.innerHTML = '<option value="">Select a model...</option>';
            
            if (data.models.length === 0) {
                select.innerHTML = '<option value="">No models found</option>';
                return;
            }
            
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.path;
                option.textContent = model.name;
                select.appendChild(option);
            });
            
            // Enable start button if model is selected
            select.addEventListener('change', () => {
                document.getElementById('start-btn').disabled = !select.value || isTrading;
            });
        })
        .catch(error => {
            addLog(`Error loading models: ${error.message}`, 'error');
        });
}

function setupEventListeners() {
    document.getElementById('start-btn').addEventListener('click', startTrading);
    document.getElementById('stop-btn').addEventListener('click', stopTrading);
}

function startTrading() {
    const modelPath = document.getElementById('model-select').value;
    const initialBalance = parseFloat(document.getElementById('initial-balance').value);
    
    if (!modelPath) {
        alert('Please select a model');
        return;
    }
    
    fetch('/api/start', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_path: modelPath,
            initial_balance: initialBalance
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLog('Trading started successfully', 'info');
            isTrading = true;
            updateControlButtons();
        } else {
            alert(`Failed to start trading: ${data.message}`);
            addLog(`Error: ${data.message}`, 'error');
        }
    })
    .catch(error => {
        alert(`Error starting trading: ${error.message}`);
        addLog(`Error: ${error.message}`, 'error');
    });
}

function stopTrading() {
    fetch('/api/stop', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addLog('Trading stopped', 'info');
            isTrading = false;
            updateControlButtons();
        } else {
            alert(`Failed to stop trading: ${data.message}`);
        }
    })
    .catch(error => {
        alert(`Error stopping trading: ${error.message}`);
    });
}

function updateControlButtons() {
    const startBtn = document.getElementById('start-btn');
    const stopBtn = document.getElementById('stop-btn');
    const modelSelect = document.getElementById('model-select');
    
    startBtn.disabled = isTrading || !modelSelect.value;
    stopBtn.disabled = !isTrading;
    modelSelect.disabled = isTrading;
    document.getElementById('initial-balance').disabled = isTrading;
}

function initializeCharts() {
    // Portfolio Value Chart
    const portfolioCtx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgb(102, 126, 234)',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Initial Balance',
                data: [],
                borderColor: 'rgb(156, 163, 175)',
                borderDash: [5, 5],
                borderWidth: 1,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
    
    // Price & Actions Chart
    const priceCtx = document.getElementById('price-chart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                tension: 0.4,
                yAxisID: 'y'
            }, {
                label: 'Buy Signals',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.5)',
                pointRadius: 5,
                pointStyle: 'triangle',
                showLine: false,
                yAxisID: 'y'
            }, {
                label: 'Sell Signals',
                data: [],
                borderColor: 'rgb(239, 68, 68)',
                backgroundColor: 'rgba(239, 68, 68, 0.5)',
                pointRadius: 5,
                pointStyle: 'triangle',
                showLine: false,
                yAxisID: 'y'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

function updateStatus(data) {
    // Update status badge
    const badge = document.getElementById('status-badge');
    badge.className = `badge badge-${data.status === 'running' ? 'running' : 'stopped'}`;
    badge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    
    // Update metrics
    document.getElementById('portfolio-value').textContent = 
        formatCurrency(data.portfolio_value || 0);
    document.getElementById('balance').textContent = 
        formatCurrency(data.balance || 0);
    document.getElementById('unrealized-pnl').textContent = 
        formatCurrency(data.unrealized_pnl || 0);
    
    const totalReturn = data.total_return || 0;
    const returnEl = document.getElementById('total-return');
    returnEl.textContent = totalReturn.toFixed(2) + '%';
    returnEl.className = 'metric-value ' + (totalReturn >= 0 ? 'positive' : 'negative');
    
    document.getElementById('current-price').textContent = 
        formatCurrency(data.current_price || 0);
    
    const position = data.position || 0;
    const positionEl = document.getElementById('position');
    if (position > 0) {
        positionEl.textContent = `LONG (${position.toFixed(4)})`;
        positionEl.style.color = '#10b981';
    } else if (position < 0) {
        positionEl.textContent = `SHORT (${Math.abs(position).toFixed(4)})`;
        positionEl.style.color = '#ef4444';
    } else {
        positionEl.textContent = 'No Position';
        positionEl.style.color = '#6b7280';
    }
    
    document.getElementById('entry-price').textContent = 
        data.entry_price ? formatCurrency(data.entry_price) : '--';
    document.getElementById('total-trades').textContent = 
        data.total_trades || 0;
    
    // Update portfolio change
    const portfolioChange = document.getElementById('portfolio-change');
    if (data.initial_balance && data.portfolio_value) {
        const change = data.portfolio_value - data.initial_balance;
        portfolioChange.textContent = 
            (change >= 0 ? '+' : '') + formatCurrency(change);
        portfolioChange.className = 'metric-change ' + 
            (change >= 0 ? 'positive' : 'negative');
    }
}

function handleTradingUpdate(data) {
    if (!data.timestamp) return;
    
    // Add to data arrays
    const timeLabel = new Date(data.timestamp).toLocaleTimeString();
    
    if (data.portfolio_value !== undefined) {
        portfolioData.push({
            x: timeLabel,
            y: data.portfolio_value
        });
        
        // Update portfolio chart
        if (portfolioChart) {
            portfolioChart.data.labels.push(timeLabel);
            portfolioChart.data.datasets[0].data.push(data.portfolio_value);
            
            // Keep only last 100 points
            if (portfolioChart.data.labels.length > 100) {
                portfolioChart.data.labels.shift();
                portfolioChart.data.datasets[0].data.shift();
                portfolioChart.data.datasets[1].data.shift();
            }
            
            portfolioChart.update('none');
        }
    }
    
    if (data.price !== undefined) {
        priceData.push({
            x: timeLabel,
            y: data.price
        });
        
        // Update price chart
        if (priceChart) {
            priceChart.data.labels.push(timeLabel);
            priceChart.data.datasets[0].data.push(data.price);
            
            // Add action markers
            if (data.action_id === 1) { // BUY
                priceChart.data.datasets[1].data.push({
                    x: timeLabel,
                    y: data.price
                });
            } else if (data.action_id === 2) { // SELL
                priceChart.data.datasets[2].data.push({
                    x: timeLabel,
                    y: data.price
                });
            }
            
            // Keep only last 100 points
            if (priceChart.data.labels.length > 100) {
                priceChart.data.labels.shift();
                priceChart.data.datasets.forEach(dataset => {
                    if (Array.isArray(dataset.data)) {
                        dataset.data.shift();
                    }
                });
            }
            
            priceChart.update('none');
        }
    }
    
    // Handle trade execution
    if (data.trade_executed && data.trade_info) {
        addTrade(data.timestamp, data.trade_info);
        addLog(`Trade executed: ${data.trade_info.type} @ ${formatCurrency(data.price)}`, 'trade');
    }
    
    // Add action to history
    if (data.action) {
        addAction(data.timestamp, data.action, data.price);
        if (data.action !== 'HOLD') {
            addLog(`Action: ${data.action} @ ${formatCurrency(data.price)}`, 'action');
        }
    }
}

function addTrade(timestamp, tradeInfo) {
    const tbody = document.getElementById('trades-tbody');
    
    // Remove empty message if exists
    if (tbody.firstChild && tbody.firstChild.classList.contains('empty')) {
        tbody.innerHTML = '';
    }
    
    const row = document.createElement('tr');
    row.innerHTML = `
        <td>${new Date(timestamp).toLocaleTimeString()}</td>
        <td>${tradeInfo.type}</td>
        <td>${formatCurrency(tradeInfo.price || 0)}</td>
        <td>${tradeInfo.size ? tradeInfo.size.toFixed(4) : '--'}</td>
        <td class="${(tradeInfo.pnl || 0) >= 0 ? 'positive' : 'negative'}">
            ${tradeInfo.pnl ? formatCurrency(tradeInfo.pnl) : '--'}
        </td>
        <td>${formatCurrency(tradeInfo.fee || 0)}</td>
    `;
    
    tbody.insertBefore(row, tbody.firstChild);
    
    // Keep only last 50 trades
    while (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }
}

function addAction(timestamp, action, price) {
    const tbody = document.getElementById('actions-tbody');
    
    // Remove empty message if exists
    if (tbody.firstChild && tbody.firstChild.classList.contains('empty')) {
        tbody.innerHTML = '';
    }
    
    const row = document.createElement('tr');
    const actionClass = action === 'BUY' ? 'positive' : action === 'SELL' ? 'negative' : '';
    row.innerHTML = `
        <td>${new Date(timestamp).toLocaleTimeString()}</td>
        <td class="${actionClass}"><strong>${action}</strong></td>
        <td>${formatCurrency(price)}</td>
    `;
    
    tbody.insertBefore(row, tbody.firstChild);
    
    // Keep only last 100 actions
    while (tbody.children.length > 100) {
        tbody.removeChild(tbody.lastChild);
    }
}

function addLog(message, type = 'info') {
    const logPanel = document.getElementById('activity-log');
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    
    const time = new Date().toLocaleTimeString();
    const typeClass = type === 'trade' ? 'log-trade' : 
                     type === 'action' ? 'log-action' : 
                     type === 'error' ? 'log-error' : 'log-message';
    
    entry.innerHTML = `
        <span class="log-time">[${time}]</span>
        <span class="${typeClass}">${message}</span>
    `;
    
    logPanel.insertBefore(entry, logPanel.firstChild);
    
    // Keep only last 100 log entries
    while (logPanel.children.length > 100) {
        logPanel.removeChild(logPanel.lastChild);
    }
}

function formatCurrency(value) {
    return '$' + parseFloat(value).toFixed(2);
}

function startPolling() {
    // Poll for status updates every 5 seconds
    setInterval(() => {
        fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'running' || data.status === 'stopped') {
                    updateStatus(data);
                    isTrading = data.status === 'running';
                    updateControlButtons();
                }
            })
            .catch(error => {
                console.error('Error polling status:', error);
            });
    }, 5000);
    
    // Load history every 10 seconds
    setInterval(() => {
        if (isTrading) {
            fetch('/api/history?limit=100')
                .then(response => response.json())
                .then(data => {
                    // Update charts with historical data if needed
                })
                .catch(error => {
                    console.error('Error loading history:', error);
                });
        }
    }, 10000);
}

