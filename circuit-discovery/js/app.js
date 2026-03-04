/**
 * Circuit Discovery Tool — Frontend
=====================================
Interactive visualization for circuit analysis results.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const modelSelect = document.getElementById('model_select');
    const modelStatus = document.getElementById('model_status');
    const analyzeBtn = document.getElementById('analyze_btn');
    const statusDiv = document.getElementById('status');
    const typeBtns = document.querySelectorAll('.type-btn');
    const promptSection = document.getElementById('prompt_section');
    const promptInput = document.getElementById('prompt_input');
    
    // Result sections
    const welcomeDiv = document.getElementById('welcome');
    const inductionResults = document.getElementById('induction_results');
    const prevTokenResults = document.getElementById('prev_token_results');
    const clusteringResults = document.getElementById('clustering_results');
    const traceResults = document.getElementById('trace_results');
    
    let currentAnalysisType = 'induction_heads';
    
    // Initialize
    checkModelStatus();
    
    // Event Listeners
    typeBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            typeBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentAnalysisType = btn.dataset.type;
            
            // Show/hide prompt section
            if (currentAnalysisType === 'trace_path') {
                promptSection.style.display = 'block';
            } else {
                promptSection.style.display = 'none';
            }
        });
    });
    
    analyzeBtn.addEventListener('click', runAnalysis);
    
    // Check API status
    async function checkModelStatus() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            const selectedModel = modelSelect.value;
            const model = data.models.find(m => m.name === selectedModel);
            
            if (model && model.loaded) {
                modelStatus.textContent = '✓ Ready';
                modelStatus.className = 'status-badge ready';
            } else {
                modelStatus.textContent = 'Not loaded (will load on demand)';
                modelStatus.className = 'status-badge loading';
            }
        } catch (e) {
            modelStatus.textContent = 'API error';
            modelStatus.className = 'status-badge error';
        }
    }
    
    modelSelect.addEventListener('change', () => {
        modelStatus.textContent = 'Checking...';
        modelStatus.className = 'status-badge loading';
        checkModelStatus();
    });
    
    // Run analysis
    async function runAnalysis() {
        const modelName = modelSelect.value;
        
        setLoading(true);
        showStatus('Loading model and running analysis... This may take a moment.', 'loading');
        
        try {
            const requestBody = {
                model_name: modelName,
                analysis_type: currentAnalysisType
            };
            
            if (currentAnalysisType === 'trace_path') {
                requestBody.prompt = promptInput.value;
                // Calculate target position (last token by default)
                const tokens = promptInput.value.split(/\s+/);
                requestBody.target_position = tokens.length - 1;
            }
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestBody)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis failed');
            }
            
            const data = await response.json();
            
            if (data.success) {
                displayResults(data);
                showStatus(`✅ Analysis complete! Found ${getResultSummary(data)}`, 'success');
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            showStatus(`❌ Error: ${error.message}`, 'error');
        } finally {
            setLoading(false);
        }
    }
    
    function getResultSummary(data) {
        switch (data.type) {
            case 'induction_heads':
                return `${data.count} induction heads`;
            case 'previous_token':
                return `${data.count} previous token heads`;
            case 'clustering':
                const types = Object.keys(data.pattern_counts).length;
                return `${types} pattern types`;
            case 'trace_path':
                return `circuit path traced`;
            default:
                return 'results ready';
        }
    }
    
    function displayResults(data) {
        // Hide all sections
        welcomeDiv.style.display = 'none';
        inductionResults.style.display = 'none';
        prevTokenResults.style.display = 'none';
        clusteringResults.style.display = 'none';
        traceResults.style.display = 'none';
        
        switch (data.type) {
            case 'induction_heads':
                displayInductionHeads(data);
                break;
            case 'previous_token':
                displayPreviousTokenHeads(data);
                break;
            case 'clustering':
                displayClustering(data);
                break;
            case 'trace_path':
                displayTracePath(data);
                break;
        }
    }
    
    function displayInductionHeads(data) {
        inductionResults.style.display = 'block';
        const grid = document.getElementById('induction_heads_grid');
        grid.innerHTML = '';
        
        if (data.heads.length === 0) {
            grid.innerHTML = '<p class="no-results">No induction heads detected in this model.</p>';
            return;
        }
        
        data.heads.forEach(head => {
            const card = document.createElement('div');
            card.className = 'head-card';
            card.innerHTML = `
                <div class="layer-label">Layer ${head.layer}</div>
                <div class="head-num">H${head.head}</div>
                <div class="score">Score: ${(head.score * 100).toFixed(1)}%</div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${head.score * 100}%"></div>
                </div>
            `;
            grid.appendChild(card);
        });
    }
    
    function displayPreviousTokenHeads(data) {
        prevTokenResults.style.display = 'block';
        const grid = document.getElementById('prev_token_grid');
        grid.innerHTML = '';
        
        if (data.heads.length === 0) {
            grid.innerHTML = '<p class="no-results">No previous token heads detected.</p>';
            return;
        }
        
        data.heads.forEach(head => {
            const card = document.createElement('div');
            card.className = 'head-card';
            card.innerHTML = `
                <div class="layer-label">Layer ${head.layer}</div>
                <div class="head-num">H${head.head}</div>
                <div class="score">Previous token</div>
            `;
            grid.appendChild(card);
        });
    }
    
    function displayClustering(data) {
        clusteringResults.style.display = 'block';
        
        // Pattern counts bar chart
        const chart = document.getElementById('pattern_chart');
        chart.innerHTML = '';
        
        const maxCount = Math.max(...Object.values(data.pattern_counts));
        
        Object.entries(data.pattern_counts)
            .sort(([,a], [,b]) => b - a)
            .forEach(([pattern, count]) => {
                const bar = document.createElement('div');
                bar.className = 'pattern-bar';
                bar.innerHTML = `
                    <div class="label">${pattern.replace('_', ' ')}</div>
                    <div class="bar-wrap">
                        <div class="bar-fill" style="width: ${(count / maxCount) * 100}%"></div>
                        <span class="count">${count}</span>
                    </div>
                `;
                chart.appendChild(bar);
            });
        
        // Head details
        const details = document.getElementById('pattern_details');
        details.innerHTML = '';
        
        data.head_details.slice(0, 20).forEach(head => {
            const card = document.createElement('div');
            card.className = 'pattern-card';
            card.innerHTML = `
                <div class="pattern-type">${head.pattern_type}</div>
                <div class="head-id">L${head.layer} H${head.head}</div>
            `;
            details.appendChild(card);
        });
    }
    
    function displayTracePath(data) {
        traceResults.style.display = 'block';
        
        // Token display
        const tokenDisplay = document.getElementById('token_display');
        tokenDisplay.innerHTML = '';
        
        data.tokens.forEach((token, idx) => {
            const span = document.createElement('span');
            span.className = 'token';
            if (idx === data.target_position) {
                span.className += ' target';
            }
            span.textContent = token.replace('Ġ', ' ').replace('Ċ', '\\n');
            tokenDisplay.appendChild(span);
        });
        
        // Path visualization
        const pathViz = document.getElementById('path_viz');
        pathViz.innerHTML = '';
        
        data.layer_contributions.forEach(layer => {
            const row = document.createElement('div');
            row.className = 'layer-row';
            
            const headsHtml = layer.top_heads.map(h => 
                `<span class="head-badge">H${h}</span>`
            ).join('');
            
            row.innerHTML = `
                <div class="layer-num">L${layer.layer}</div>
                <div class="heads">${headsHtml}</div>
            `;
            
            pathViz.appendChild(row);
        });
    }
    
    // UI Helpers
    function setLoading(loading) {
        analyzeBtn.disabled = loading;
        const btnText = analyzeBtn.querySelector('.btn-text');
        const btnLoading = analyzeBtn.querySelector('.btn-loading');
        
        if (loading) {
            btnText.style.display = 'none';
            btnLoading.style.display = 'inline';
        } else {
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    }
    
    function showStatus(message, type) {
        statusDiv.textContent = message;
        statusDiv.className = `status ${type} show`;
    }
    
    // Poll model status periodically
    setInterval(checkModelStatus, 30000);  // Every 30 seconds
});
