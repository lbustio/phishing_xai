document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const subjectInput = document.getElementById('subjectInput');
    const loader = document.getElementById('loader');
    const loaderMessage = document.getElementById('loaderMessage');
    const progressBar = document.querySelector('.progress');
    const resultsSection = document.getElementById('resultsSection');
    
    // UI Elements map
    const xaiPanel = document.getElementById('xaiPanel');
    const threatIcon = document.getElementById('threatIcon');
    const threatTitle = document.getElementById('threatTitle');
    const threatConfidence = document.getElementById('threatConfidence');
    const naturalExplanation = document.getElementById('naturalExplanation');
    const semanticReasoningText = document.getElementById('semanticReasoningText');
    const keywordChips = document.getElementById('keywordChips');
    const emailSubjectPrev = document.getElementById('emailSubjectPrev');
    const emailBodySim = document.getElementById('emailBodySim');
    const semanticPanel = document.getElementById('semanticPanel');
    const semanticPlot = document.getElementById('semanticPlot');
    const semanticInterpretation = document.getElementById('semanticInterpretation');
    const semanticNote = document.getElementById('semanticNote');
    const nearestPhishingMetric = document.getElementById('nearestPhishingMetric');
    const nearestLegitimateMetric = document.getElementById('nearestLegitimateMetric');
    const centroidMetrics = document.getElementById('centroidMetrics');
    const semanticNeighborSummary = document.getElementById('semanticNeighborSummary');
    const hoverDetails = document.getElementById('hoverDetails');
    const semanticDebug = document.getElementById('semanticDebug');
    const neighborCountInput = document.getElementById('neighborCount');
    const neighborCountValue = document.getElementById('neighborCountValue');
    const semanticModeBtn = document.getElementById('semanticModeBtn');

    const logTerminal = document.getElementById('logTerminal');
    const logContent = document.getElementById('logContent');

    const resultActions = document.getElementById('resultActions');
    const exportPdfBtn = document.getElementById('exportPdfBtn');

    const historySidebar = document.getElementById('historySidebar');
    const historyList = document.getElementById('historyList');
    const openHistoryBtn = document.getElementById('openHistory');
    const closeHistoryBtn = document.getElementById('closeHistory');
    const historyModalOverlay = document.getElementById('historyModalOverlay');
    const closeHistoryModalBtn = document.getElementById('closeHistoryModal');
    const exportHistoryPdfBtn = document.getElementById('exportHistoryPdfBtn');

    const historyView = {
        xaiPanel: document.getElementById('historyXaiPanel'),
        threatIcon: document.getElementById('historyThreatIcon'),
        threatTitle: document.getElementById('historyThreatTitle'),
        threatConfidence: document.getElementById('historyThreatConfidence'),
        naturalExplanation: document.getElementById('historyNaturalExplanation'),
        semanticReasoningText: document.getElementById('historySemanticReasoningText'),
        keywordChips: document.getElementById('historyKeywordChips'),
        emailSubjectPrev: document.getElementById('historyEmailSubjectPrev'),
        emailBodySim: document.getElementById('historyEmailBodySim'),
    };

    const historySemanticView = {
        panel: document.getElementById('historySemanticPanel'),
        plot: document.getElementById('historySemanticPlot'),
        interpretation: document.getElementById('historySemanticInterpretation'),
        note: document.getElementById('historySemanticNote'),
        nearestPhishingMetric: document.getElementById('historyNearestPhishingMetric'),
        nearestLegitimateMetric: document.getElementById('historyNearestLegitimateMetric'),
        centroidMetrics: document.getElementById('historyCentroidMetrics'),
        neighborSummary: document.getElementById('historySemanticNeighborSummary'),
        hoverDetails: document.getElementById('historyHoverDetails'),
        debug: document.getElementById('historySemanticDebug'),
    };

    const soundToggle = document.getElementById('soundToggleFloating') || document.getElementById('soundToggle');
    const soundIcon = document.getElementById('soundIconFloating') || document.getElementById('soundIcon');
    const dynamicTechStack = document.getElementById('dynamicTechStack');
    const footerTechStack = document.querySelector('.footer-stack');
    let runtimeTechnologies = [];
    
    // --- Config Loading ---
    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            const config = await res.json();
            if (dynamicTechStack) {
                dynamicTechStack.innerHTML = `
                    <span class="pipeline-part">Pipeline: <strong>${config.embedding}</strong></span> | 
                    <span class="pipeline-part">Clasificador: <strong>${config.classifier}</strong></span> | 
                    <span class="pipeline-part">XAI: <strong>Leave-One-Out</strong></span>
                `;
            }
            runtimeTechnologies = Array.isArray(config.technologies_used)
                ? [...config.technologies_used]
                : [];
            if (footerTechStack && runtimeTechnologies.length > 0) {
                footerTechStack.innerHTML = runtimeTechnologies
                    .map(item => `<span class="footer-pill">${item}</span>`)
                    .join('');
            }
        } catch (err) {
            console.error("Failed to load config", err);
        }
    }
    loadConfig();

    // Audio Engine using Web Audio API
    class AudioEngine {
        constructor() {
            this.ctx = null;
            this.muted = localStorage.getItem('xai_muted') === 'true';
            this.updateUI();
        }

        init() {
            if (!this.ctx) {
                this.ctx = new (window.AudioContext || window.webkitAudioContext)();
            }
        }

        toggle() {
            this.muted = !this.muted;
            localStorage.setItem('xai_muted', this.muted);
            this.updateUI();
        }

        updateUI() {
            if (this.muted) {
                soundToggle.classList.add('muted');
                soundIcon.textContent = '🔇';
            } else {
                soundToggle.classList.remove('muted');
                soundIcon.textContent = '🔊';
            }
        }

        playAlarm() {
            if (this.muted) return;
            this.init();
            const now = this.ctx.currentTime;
            for(let i=0; i<3; i++) {
                const t = now + (i * 0.4);
                this.beep(880, t, 0.2, 'sawtooth');
                this.beep(660, t + 0.2, 0.2, 'sawtooth');
            }
        }

        playSafe() {
            if (this.muted) return;
            this.init();
            const now = this.ctx.currentTime;
            this.beep(440, now, 0.4, 'sine', 0.1);
            this.beep(554.37, now + 0.1, 0.4, 'sine', 0.08);
            this.beep(659.25, now + 0.2, 0.4, 'sine', 0.06);
        }

        beep(freq, start, duration, type, volume = 0.2) {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, start);
            gain.gain.setValueAtTime(0, start);
            gain.gain.linearRampToValueAtTime(volume, start + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.01, start + duration);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(start);
            osc.stop(start + duration);
        }
    }

    const audio = new AudioEngine();

    soundToggle.addEventListener('click', () => {
        audio.toggle();
    });

    analyzeBtn.addEventListener('click', () => {
        audio.init(); 
        analyzeSubject();
    });
    subjectInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            audio.init();
            analyzeSubject();
        }
    });

    const cancelBtn = document.getElementById('cancelBtn');
    let analysisController = null;
    let lastResult = null;
    let activeHistoryResult = null;

    cancelBtn.addEventListener('click', () => {
        if (analysisController) {
            analysisController.abort();
            addLogEntry("🛑 Análisis cancelado por el usuario.");
        }
    });

    async function analyzeSubject() {
        const subject = subjectInput.value.trim();
        if (!subject) return;

        analysisController = new AbortController();
        const signal = analysisController.signal;

        resultsSection.classList.add('hidden');
        resultActions.classList.add('hidden');
        logTerminal.classList.remove('hidden');
        logContent.innerHTML = '';
        analyzeBtn.disabled = true;
        setSemanticDebug('Solicitud iniciada. Esperando respuesta del backend para el mapa semántico.');
        
        loader.classList.remove('hidden');
        progressBar.style.width = '0%';
        loaderMessage.textContent = "Iniciando conexión...";

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject }),
                signal: signal
            });

            if(!response.ok) throw new Error("Error en el servidor");
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let resultData = null;
            let streamBuffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    const tail = streamBuffer.trim();
                    if (tail) {
                        try {
                            const payload = JSON.parse(tail);
                            if (payload.type === 'log') {
                                addLogEntry(payload.content);
                            } else if (payload.type === 'result') {
                                resultData = payload.data;
                                const pointCount = resultData?.semantic_map?.points?.length;
                                if (pointCount) {
                                    setSemanticDebug(`Respuesta recibida al cerrar stream. El backend entregó ${pointCount} puntos semánticos.`);
                                }
                            }
                        } catch (e) {
                            console.error('Error parsing final buffered stream chunk:', e, tail.slice(0, 400));
                            setSemanticDebug('El stream terminó con un fragmento JSON incompleto o inválido.');
                        }
                    }
                    break;
                }
                
                streamBuffer += decoder.decode(value, { stream: true });
                const lines = streamBuffer.split('\n');
                streamBuffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const payload = JSON.parse(line);
                        if (payload.type === 'log') {
                            addLogEntry(payload.content);
                            if ((payload.content || '').toLowerCase().includes('mapa semántico')
                                || (payload.content || '').toLowerCase().includes('payload semántico')
                                || (payload.content || '').toLowerCase().includes('frontend')) {
                                setSemanticDebug(`Backend: ${payload.content}`);
                            }
                        } else if (payload.type === 'result') {
                            resultData = payload.data;
                            const pointCount = resultData?.semantic_map?.points?.length;
                            if (pointCount) {
                                setSemanticDebug(`Respuesta recibida. El backend entregó ${pointCount} puntos semánticos.`);
                            } else if (resultData?.semantic_error) {
                                setSemanticDebug(`Respuesta recibida sin mapa. Error reportado: ${resultData.semantic_error}`);
                            } else {
                                setSemanticDebug('Respuesta recibida sin mapa semántico y sin detalle adicional.');
                            }
                        }
                    } catch (e) {
                        console.error("Error parsing stream chunk:", e, line.slice(0, 400));
                        setSemanticDebug('Se detectó un fragmento NDJSON inválido durante la recepción del análisis.');
                    }
                }
            }
            
            progressBar.style.width = '100%';
            loaderMessage.textContent = "Análisis Completado";
            
            if (resultData) {
                setTimeout(() => {
                    loader.classList.add('hidden');
                    displayResults(resultData);
                    analyzeBtn.disabled = false;
                }, 800);
            }

        } catch (error) {
            if (error.name === 'AbortError') {
                loaderMessage.textContent = "Análisis Cancelado.";
                progressBar.style.background = "#6b7280";
            } else {
                addLogEntry("❌ ERROR: " + error.message);
                loaderMessage.textContent = "Error de conexión.";
                progressBar.style.background = "#ef4444";
            }
            
            setTimeout(() => { 
                analyzeBtn.disabled = false; 
                loader.classList.add('hidden');
            }, 1500);
            console.error(error);
        } finally {
            analysisController = null;
        }
    }

    function addLogEntry(text) {
        const now = new Date();
        const ts = now.getFullYear()
            + '-' + String(now.getMonth() + 1).padStart(2, '0')
            + '-' + String(now.getDate()).padStart(2, '0')
            + ' ' + String(now.getHours()).padStart(2, '0')
            + ':' + String(now.getMinutes()).padStart(2, '0')
            + ':' + String(now.getSeconds()).padStart(2, '0');

        const entry = document.createElement('div');
        entry.className = 'log-entry';

        const tsSpan = document.createElement('span');
        tsSpan.className = 'log-ts';
        tsSpan.textContent = ts;

        const msgSpan = document.createElement('span');
        msgSpan.className = 'log-msg';
        msgSpan.textContent = ` > ${text}`;

        entry.appendChild(tsSpan);
        entry.appendChild(msgSpan);
        logContent.appendChild(entry);

        // Auto-scroll only if user is already near the bottom
        const atBottom = logContent.scrollHeight - logContent.scrollTop - logContent.clientHeight < 60;
        if (atBottom) {
            logContent.scrollTop = logContent.scrollHeight;
        }

        // Sync loader message with latest log
        loaderMessage.textContent = text;
    }

    const masterReasoningPanel = document.getElementById('masterReasoningPanel');
    const masterExplanation = document.getElementById('masterExplanation');
    let semanticState = {
        data: null,
        neighborCount: Number(neighborCountInput?.value || 5),
        neighborsOnly: false,
    };
    let historySemanticState = {
        data: null,
        neighborCount: 5,
        neighborsOnly: false,
    };

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatMetric(value, digits = 3) {
        const num = Number(value);
        return Number.isFinite(num) ? num.toFixed(digits) : 'N/D';
    }

    function buildSemanticNarrative(semanticMap) {
        if (!semanticMap) return '';

        const neighborSummary = semanticMap.neighbor_summary || {};
        const dominant = neighborSummary.dominant_label === 'phishing' ? 'phishing' : 'legítimos';
        const nearestPhishing = Number(semanticMap?.nearest_by_class?.phishing?.distance);
        const nearestLegitimate = Number(semanticMap?.nearest_by_class?.legitimate?.distance);
        const centroidPhishing = Number(semanticMap?.centroid_distances?.phishing);
        const centroidLegitimate = Number(semanticMap?.centroid_distances?.legitimate);

        const localClause = `En el mapa semántico, el asunto cae en un vecindario dominado por ejemplos ${dominant}.`;
        const neighborClause = Number.isFinite(neighborSummary.k)
            ? ` Entre sus ${neighborSummary.k} vecinos más cercanos, ${neighborSummary.phishing ?? 0} son phishing y ${neighborSummary.legitimate ?? 0} son legítimos.`
            : '';

        let nearestClause = '';
        if (Number.isFinite(nearestPhishing) && Number.isFinite(nearestLegitimate)) {
            nearestClause = nearestPhishing <= nearestLegitimate
                ? ` El punto phishing más cercano queda a distancia coseno ${formatMetric(nearestPhishing)}, menor que la distancia al punto legítimo más cercano (${formatMetric(nearestLegitimate)}).`
                : ` El punto legítimo más cercano queda a distancia coseno ${formatMetric(nearestLegitimate)}, menor que la distancia al punto phishing más cercano (${formatMetric(nearestPhishing)}).`;
        }

        let centroidClause = '';
        if (Number.isFinite(centroidPhishing) && Number.isFinite(centroidLegitimate)) {
            centroidClause = centroidPhishing <= centroidLegitimate
                ? ` También queda más próximo al centroide phishing (${formatMetric(centroidPhishing)}) que al centroide legítimo (${formatMetric(centroidLegitimate)}).`
                : ` También queda más próximo al centroide legítimo (${formatMetric(centroidLegitimate)}) que al centroide phishing (${formatMetric(centroidPhishing)}).`;
        }

        return `${localClause}${neighborClause}${nearestClause}${centroidClause}`.trim();
    }

    function setSemanticDebug(message, targetDebug = semanticDebug) {
        if (targetDebug) {
            targetDebug.textContent = message;
        }
        console.info(`[semantic] ${message}`);
    }

    function getPointByIndex(points, index) {
        return Array.isArray(points) ? points.find(point => point.id === index) : null;
    }

    function updateHoverCard(point, targetHoverDetails = hoverDetails) {
        if (!targetHoverDetails) return;

        if (!point) {
            targetHoverDetails.textContent = 'Pasa el mouse sobre un punto para ver el asunto, su clase y su semejanza con el asunto analizado.';
            return;
        }

        const label = point.label === 'phishing' ? 'Phishing' : 'Legítimo';
        const rank = point.neighbor_rank > 0 ? ` | vecino #${point.neighbor_rank}` : '';
        targetHoverDetails.innerHTML = `
            <strong>${label}</strong> | similitud coseno ${formatMetric(point.similarity)} | distancia ${formatMetric(point.distance)}${rank}
            <br>
            <span>${escapeHtml(point.subject_preview || point.subject || '')}</span>
        `;
    }

    function buildPointHoverText(point) {
        const label = point.label === 'phishing' ? 'Phishing' : 'Legítimo';
        const rank = point.neighbor_rank > 0 ? `<br>Rango local: #${point.neighbor_rank}` : '';
        return `<b>${label}</b><br>${escapeHtml(point.subject_preview || point.subject || '')}<br>Similitud coseno: ${formatMetric(point.similarity)}<br>Distancia coseno: ${formatMetric(point.distance)}${rank}`;
    }

    function renderSemanticMap(semanticMap, semanticError = null) {
        const isNewPayload = semanticState.data !== semanticMap;
        semanticState.data = semanticMap || null;
        if (!semanticMap) {
            semanticPanel.classList.remove('hidden');
            setSemanticDebug(semanticError ? `No renderizado. Error del backend: ${semanticError}` : 'No renderizado. El backend no devolvió semantic_map.');
            semanticInterpretation.textContent = 'No fue posible construir el mapa semántico interactivo para esta ejecución.';
            semanticNote.textContent = semanticError
                ? `Detalle del backend: ${semanticError}`
                : 'El backend no devolvió datos semánticos. Revisa que el dataset fuente del run y sus artefactos estén disponibles localmente.';
            semanticPlot.innerHTML = '<div class="semantic-placeholder">El servidor no pudo generar el scatter semántico con datos reales.</div>';
            semanticNeighborSummary.textContent = 'Sin datos.';
            nearestPhishingMetric.textContent = 'Sin datos.';
            nearestLegitimateMetric.textContent = 'Sin datos.';
            centroidMetrics.textContent = 'Sin datos.';
            updateHoverCard(null);
            console.warn('semantic_map ausente en la respuesta del análisis.');
            return;
        }
        if (!window.Plotly) {
            semanticPanel.classList.remove('hidden');
            setSemanticDebug('No renderizado. Plotly no está disponible en window.');
            semanticInterpretation.textContent = semanticMap.interpretation || 'La visualización semántica no pudo renderizarse en el navegador.';
            semanticNote.textContent = 'Los datos del mapa sí llegaron, pero Plotly no se cargó. Esto suele indicar que el CDN fue bloqueado o no estuvo disponible.';
            semanticPlot.innerHTML = '<div class="semantic-placeholder">Plotly no se cargó en este navegador. El panel semántico requiere ese recurso para dibujar el scatter interactivo.</div>';
            console.warn('Plotly no está disponible en window.');
            return;
        }

        semanticPanel.classList.remove('hidden');
        setSemanticDebug(
            `Renderizando scatter con ${Array.isArray(semanticMap.points) ? semanticMap.points.length : 0} puntos, ${semanticState.neighborCount} vecinos resaltados y modo ${semanticState.neighborsOnly ? 'solo vecinos' : 'completo'}.`
        );
        semanticInterpretation.textContent = semanticMap.interpretation || 'Visualización semántica disponible.';
        semanticNote.textContent = `${semanticMap.note || ''} Vecinos resaltados activos: ${semanticState.neighborCount}.`;

        const maxNeighbors = Number(semanticMap.max_neighbor_count || 5);
        const defaultNeighbors = Number(semanticMap.default_neighbor_count || Math.min(5, maxNeighbors));
        neighborCountInput.max = String(Math.max(3, maxNeighbors));
        if (isNewPayload) {
            semanticState.neighborCount = Math.min(defaultNeighbors, maxNeighbors);
            semanticState.neighborsOnly = false;
        } else {
            semanticState.neighborCount = Math.min(Math.max(3, semanticState.neighborCount), maxNeighbors);
        }
        neighborCountInput.value = String(semanticState.neighborCount);
        neighborCountValue.textContent = neighborCountInput.value;
        semanticModeBtn.textContent = semanticState.neighborsOnly ? 'Mostrar todos los puntos' : 'Mostrar solo vecinos';

        const points = semanticMap.points || [];
        const neighbors = points.filter(point => point.neighbor_rank > 0 && point.neighbor_rank <= semanticState.neighborCount);
        const allPhishing = points.filter(point => point.label === 'phishing');
        const allLegitimate = points.filter(point => point.label === 'legitimate');
        const neighborIds = new Set(neighbors.map(point => point.id));
        const basePhishing = allPhishing.filter(point => !neighborIds.has(point.id));
        const baseLegitimate = allLegitimate.filter(point => !neighborIds.has(point.id));
        const phishingNeighbors = neighbors.filter(point => point.label === 'phishing');
        const legitimateNeighbors = neighbors.filter(point => point.label === 'legitimate');
        const neighborLineX = [];
        const neighborLineY = [];
        neighbors.forEach((point) => {
            neighborLineX.push(semanticMap.analysis_point.x, point.x, null);
            neighborLineY.push(semanticMap.analysis_point.y, point.y, null);
        });

        const traces = [];
        if (!semanticState.neighborsOnly) {
            traces.push({
                type: 'scattergl',
                mode: 'markers',
                name: 'Phishing',
                x: basePhishing.map(point => point.x),
                y: basePhishing.map(point => point.y),
                customdata: basePhishing,
                text: basePhishing.map(buildPointHoverText),
                marker: {
                    color: 'rgba(239, 68, 68, 0.16)',
                    size: 7,
                    line: { color: 'rgba(255,255,255,0.06)', width: 0.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            });
            traces.push({
                type: 'scattergl',
                mode: 'markers',
                name: 'Legítimos',
                x: baseLegitimate.map(point => point.x),
                y: baseLegitimate.map(point => point.y),
                customdata: baseLegitimate,
                text: baseLegitimate.map(buildPointHoverText),
                marker: {
                    color: 'rgba(16, 185, 129, 0.14)',
                    size: 7,
                    line: { color: 'rgba(255,255,255,0.06)', width: 0.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            });
        }

        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: 'Conexiones locales',
            x: neighborLineX,
            y: neighborLineY,
            line: {
                color: 'rgba(250, 204, 21, 0.42)',
                width: 1.4,
            },
            hoverinfo: 'skip',
            showlegend: false,
        });

        traces.push({
            type: 'scattergl',
            mode: 'markers',
            name: 'Vecinos phishing',
            x: phishingNeighbors.map(point => point.x),
            y: phishingNeighbors.map(point => point.y),
            customdata: phishingNeighbors,
            text: phishingNeighbors.map(buildPointHoverText),
            marker: {
                color: 'rgba(239, 68, 68, 0.96)',
                size: 17,
                symbol: 'diamond',
                line: { color: 'rgba(255,255,255,0.85)', width: 1.4 },
            },
            hovertemplate: '%{text}<extra></extra>',
        });
        traces.push({
            type: 'scattergl',
            mode: 'markers',
            name: 'Vecinos legítimos',
            x: legitimateNeighbors.map(point => point.x),
            y: legitimateNeighbors.map(point => point.y),
            customdata: legitimateNeighbors,
            text: legitimateNeighbors.map(buildPointHoverText),
            marker: {
                color: 'rgba(16, 185, 129, 0.96)',
                size: 17,
                symbol: 'diamond',
                line: { color: 'rgba(255,255,255,0.85)', width: 1.4 },
            },
            hovertemplate: '%{text}<extra></extra>',
        });
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Centroide phishing',
            x: [semanticMap.centroids.phishing.x],
            y: [semanticMap.centroids.phishing.y],
            customdata: [{ label: 'phishing-centroid' }],
            marker: {
                symbol: 'x',
                color: 'rgba(255, 153, 153, 0.95)',
                size: 14,
                line: { width: 2, color: 'rgba(255,255,255,0.8)' },
            },
            hovertemplate: '<b>Centroide phishing</b><extra></extra>',
        });
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Centroide legítimo',
            x: [semanticMap.centroids.legitimate.x],
            y: [semanticMap.centroids.legitimate.y],
            customdata: [{ label: 'legitimate-centroid' }],
            marker: {
                symbol: 'x',
                color: 'rgba(167, 243, 208, 0.95)',
                size: 14,
                line: { width: 2, color: 'rgba(255,255,255,0.8)' },
            },
            hovertemplate: '<b>Centroide legítimo</b><extra></extra>',
        });
        traces.push({
            type: 'scatter',
            mode: 'markers',
            name: 'Halo asunto analizado',
            x: [semanticMap.analysis_point.x],
            y: [semanticMap.analysis_point.y],
            marker: {
                color: 'rgba(59, 130, 246, 0.24)',
                size: 24,
                line: { color: 'rgba(96, 165, 250, 0.78)', width: 2.2 },
            },
            hoverinfo: 'skip',
            showlegend: false,
        });
        traces.push({
            type: 'scatter',
            mode: 'markers+text',
            name: 'Asunto analizado',
            x: [semanticMap.analysis_point.x],
            y: [semanticMap.analysis_point.y],
            text: ['ASUNTO ANALIZADO'],
            textposition: 'top center',
            textfont: {
                color: '#93c5fd',
                size: 13,
                family: 'Inter, sans-serif',
            },
            marker: {
                color: '#2563eb',
                size: 14,
                line: { color: '#dbeafe', width: 2.4 },
            },
            hovertemplate: `<b>Asunto analizado</b><br>${escapeHtml(semanticMap.analysis_point.subject_preview || semanticMap.analysis_point.subject || '')}<extra></extra>`,
        });
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: 'Al centroide phishing',
            x: [semanticMap.analysis_point.x, semanticMap.centroids.phishing.x],
            y: [semanticMap.analysis_point.y, semanticMap.centroids.phishing.y],
            line: { color: 'rgba(255, 120, 120, 0.9)', width: 2.5, dash: 'dash' },
            hoverinfo: 'skip',
            showlegend: false,
        });
        traces.push({
            type: 'scatter',
            mode: 'lines',
            name: 'Al centroide legítimo',
            x: [semanticMap.analysis_point.x, semanticMap.centroids.legitimate.x],
            y: [semanticMap.analysis_point.y, semanticMap.centroids.legitimate.y],
            line: { color: 'rgba(110, 231, 183, 0.9)', width: 2.5, dash: 'dash' },
            hoverinfo: 'skip',
            showlegend: false,
        });

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(4, 9, 18, 0.82)',
            font: { color: '#e2e8f0', family: 'Inter, sans-serif' },
            margin: { l: 48, r: 18, t: 20, b: 44 },
            hovermode: 'closest',
            legend: {
                orientation: 'h',
                y: 1.08,
                x: 0,
                bgcolor: 'rgba(0,0,0,0)',
            },
            xaxis: {
                title: 'Componente principal 1',
                gridcolor: 'rgba(148,163,184,0.1)',
                zerolinecolor: 'rgba(148,163,184,0.16)',
            },
            yaxis: {
                title: 'Componente principal 2',
                gridcolor: 'rgba(148,163,184,0.1)',
                zerolinecolor: 'rgba(148,163,184,0.16)',
            },
        };

        const config = {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        };

        Plotly.newPlot(semanticPlot, traces, layout, config);
        setSemanticDebug(`Scatter renderizado correctamente con ${traces.length} capas visuales.`);

        semanticNeighborSummary.textContent = `Entre los ${semanticState.neighborCount} vecinos más cercanos, ${neighbors.filter(point => point.label === 'phishing').length} son phishing y ${neighbors.filter(point => point.label === 'legitimate').length} son legítimos.`;

        const nearestPhishing = getPointByIndex(points, semanticMap.nearest_by_class?.phishing?.index);
        const nearestLegitimate = getPointByIndex(points, semanticMap.nearest_by_class?.legitimate?.index);
        nearestPhishingMetric.innerHTML = nearestPhishing
            ? `Distancia ${formatMetric(semanticMap.nearest_by_class.phishing.distance)} | similitud ${formatMetric(semanticMap.nearest_by_class.phishing.similarity)}<br><span>${escapeHtml(nearestPhishing.subject_preview || nearestPhishing.subject || '')}</span>`
            : 'Sin datos.';
        nearestLegitimateMetric.innerHTML = nearestLegitimate
            ? `Distancia ${formatMetric(semanticMap.nearest_by_class.legitimate.distance)} | similitud ${formatMetric(semanticMap.nearest_by_class.legitimate.similarity)}<br><span>${escapeHtml(nearestLegitimate.subject_preview || nearestLegitimate.subject || '')}</span>`
            : 'Sin datos.';
        centroidMetrics.innerHTML = `
            Phishing: ${formatMetric(semanticMap.centroid_distances?.phishing)}<br>
            Legítimo: ${formatMetric(semanticMap.centroid_distances?.legitimate)}
        `;
        updateHoverCard(null);

        if (typeof semanticPlot.removeAllListeners === 'function') {
            semanticPlot.removeAllListeners('plotly_hover');
            semanticPlot.removeAllListeners('plotly_unhover');
        }
        semanticPlot.on('plotly_hover', (event) => {
            const point = event?.points?.[0]?.customdata;
            if (point && (point.subject_preview || point.subject)) {
                updateHoverCard(point);
            }
        });

        semanticPlot.on('plotly_unhover', () => {
            updateHoverCard(null);
        });
    }

    function applySemanticReasoning(target, semanticMap) {
        if (!target?.semanticReasoningText) return;

        const narrative = buildSemanticNarrative(semanticMap);
        if (!narrative) {
            target.semanticReasoningText.classList.add('hidden');
            target.semanticReasoningText.innerHTML = '';
            return;
        }

        target.semanticReasoningText.classList.remove('hidden');
        target.semanticReasoningText.innerHTML = `<strong>Lectura del mapa semántico:</strong> ${escapeHtml(narrative)}`;
    }

    function renderHistorySemanticMap(semanticMap, semanticError = null) {
        const isNewPayload = historySemanticState.data !== semanticMap;
        historySemanticState.data = semanticMap || null;

        if (!semanticMap) {
            historySemanticView.panel.classList.remove('hidden');
            setSemanticDebug(
                semanticError ? `Historial: error del backend: ${semanticError}` : 'Historial: no se recibió semantic_map.',
                historySemanticView.debug,
            );
            historySemanticView.interpretation.textContent = 'No fue posible reconstruir el mapa semántico de este análisis archivado.';
            historySemanticView.note.textContent = semanticError
                ? `Detalle del backend: ${semanticError}`
                : 'No fue posible regenerar el scatter real para este análisis archivado.';
            historySemanticView.plot.innerHTML = '<div class="semantic-placeholder">El historial no pudo reconstruir el scatter semántico.</div>';
            historySemanticView.neighborSummary.textContent = 'Sin datos.';
            historySemanticView.nearestPhishingMetric.textContent = 'Sin datos.';
            historySemanticView.nearestLegitimateMetric.textContent = 'Sin datos.';
            historySemanticView.centroidMetrics.textContent = 'Sin datos.';
            updateHoverCard(null, historySemanticView.hoverDetails);
            return;
        }

        historySemanticView.panel.classList.remove('hidden');
        historySemanticView.interpretation.textContent = semanticMap.interpretation || 'Visualización semántica disponible.';
        historySemanticView.note.textContent = semanticMap.note || '';

        const maxNeighbors = Number(semanticMap.max_neighbor_count || 5);
        const defaultNeighbors = Number(semanticMap.default_neighbor_count || Math.min(5, maxNeighbors));
        historySemanticState.neighborCount = isNewPayload
            ? Math.min(defaultNeighbors, maxNeighbors)
            : Math.min(Math.max(3, historySemanticState.neighborCount), maxNeighbors);
        historySemanticState.neighborsOnly = false;

        const points = semanticMap.points || [];
        const neighbors = points.filter(point => point.neighbor_rank > 0 && point.neighbor_rank <= historySemanticState.neighborCount);
        const allPhishing = points.filter(point => point.label === 'phishing');
        const allLegitimate = points.filter(point => point.label === 'legitimate');
        const neighborIds = new Set(neighbors.map(point => point.id));
        const basePhishing = allPhishing.filter(point => !neighborIds.has(point.id));
        const baseLegitimate = allLegitimate.filter(point => !neighborIds.has(point.id));
        const phishingNeighbors = neighbors.filter(point => point.label === 'phishing');
        const legitimateNeighbors = neighbors.filter(point => point.label === 'legitimate');
        const neighborLineX = [];
        const neighborLineY = [];
        neighbors.forEach((point) => {
            neighborLineX.push(semanticMap.analysis_point.x, point.x, null);
            neighborLineY.push(semanticMap.analysis_point.y, point.y, null);
        });

        const traces = [
            {
                type: 'scattergl',
                mode: 'markers',
                name: 'Phishing',
                x: basePhishing.map(point => point.x),
                y: basePhishing.map(point => point.y),
                customdata: basePhishing,
                text: basePhishing.map(buildPointHoverText),
                marker: {
                    color: 'rgba(239, 68, 68, 0.16)',
                    size: 7,
                    line: { color: 'rgba(255,255,255,0.06)', width: 0.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            },
            {
                type: 'scatter',
                mode: 'lines',
                name: 'Conexiones locales',
                x: neighborLineX,
                y: neighborLineY,
                line: {
                    color: 'rgba(250, 204, 21, 0.42)',
                    width: 1.4,
                },
                hoverinfo: 'skip',
                showlegend: false,
            },
            {
                type: 'scattergl',
                mode: 'markers',
                name: 'Legítimos',
                x: baseLegitimate.map(point => point.x),
                y: baseLegitimate.map(point => point.y),
                customdata: baseLegitimate,
                text: baseLegitimate.map(buildPointHoverText),
                marker: {
                    color: 'rgba(16, 185, 129, 0.14)',
                    size: 7,
                    line: { color: 'rgba(255,255,255,0.06)', width: 0.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            },
            {
                type: 'scattergl',
                mode: 'markers',
                name: 'Vecinos phishing',
                x: phishingNeighbors.map(point => point.x),
                y: phishingNeighbors.map(point => point.y),
                customdata: phishingNeighbors,
                text: phishingNeighbors.map(buildPointHoverText),
                marker: {
                    color: 'rgba(239, 68, 68, 0.96)',
                    size: 17,
                    symbol: 'diamond',
                    line: { color: 'rgba(255,255,255,0.85)', width: 1.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            },
            {
                type: 'scattergl',
                mode: 'markers',
                name: 'Vecinos legítimos',
                x: legitimateNeighbors.map(point => point.x),
                y: legitimateNeighbors.map(point => point.y),
                customdata: legitimateNeighbors,
                text: legitimateNeighbors.map(buildPointHoverText),
                marker: {
                    color: 'rgba(16, 185, 129, 0.96)',
                    size: 17,
                    symbol: 'diamond',
                    line: { color: 'rgba(255,255,255,0.85)', width: 1.4 },
                },
                hovertemplate: '%{text}<extra></extra>',
            },
            {
                type: 'scatter',
                mode: 'markers',
                name: 'Centroide phishing',
                x: [semanticMap.centroids.phishing.x],
                y: [semanticMap.centroids.phishing.y],
                marker: {
                    symbol: 'x',
                    color: 'rgba(255, 153, 153, 0.95)',
                    size: 14,
                    line: { width: 2, color: 'rgba(255,255,255,0.8)' },
                },
                hovertemplate: '<b>Centroide phishing</b><extra></extra>',
            },
            {
                type: 'scatter',
                mode: 'markers',
                name: 'Centroide legítimo',
                x: [semanticMap.centroids.legitimate.x],
                y: [semanticMap.centroids.legitimate.y],
                marker: {
                    symbol: 'x',
                    color: 'rgba(167, 243, 208, 0.95)',
                    size: 14,
                    line: { width: 2, color: 'rgba(255,255,255,0.8)' },
                },
                hovertemplate: '<b>Centroide legítimo</b><extra></extra>',
            },
            {
                type: 'scatter',
                mode: 'markers',
                name: 'Halo asunto analizado',
                x: [semanticMap.analysis_point.x],
                y: [semanticMap.analysis_point.y],
                marker: {
                    color: 'rgba(59, 130, 246, 0.24)',
                    size: 40,
                    line: { color: 'rgba(96, 165, 250, 0.78)', width: 2.2 },
                },
                hoverinfo: 'skip',
                showlegend: false,
            },
            {
                type: 'scatter',
                mode: 'markers+text',
                name: 'Asunto analizado',
                x: [semanticMap.analysis_point.x],
                y: [semanticMap.analysis_point.y],
                text: ['ASUNTO ANALIZADO'],
                textposition: 'top center',
                textfont: {
                    color: '#93c5fd',
                    size: 13,
                    family: 'Inter, sans-serif',
                },
                marker: {
                    color: '#2563eb',
                    size: 24,
                    line: { color: '#dbeafe', width: 3.4 },
                },
                hovertemplate: `<b>Asunto analizado</b><br>${escapeHtml(semanticMap.analysis_point.subject_preview || semanticMap.analysis_point.subject || '')}<extra></extra>`,
            },
            {
                type: 'scatter',
                mode: 'lines',
                name: 'Al centroide phishing',
                x: [semanticMap.analysis_point.x, semanticMap.centroids.phishing.x],
                y: [semanticMap.analysis_point.y, semanticMap.centroids.phishing.y],
                line: { color: 'rgba(255, 120, 120, 0.9)', width: 2.5, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: false,
            },
            {
                type: 'scatter',
                mode: 'lines',
                name: 'Al centroide legítimo',
                x: [semanticMap.analysis_point.x, semanticMap.centroids.legitimate.x],
                y: [semanticMap.analysis_point.y, semanticMap.centroids.legitimate.y],
                line: { color: 'rgba(110, 231, 183, 0.9)', width: 2.5, dash: 'dash' },
                hoverinfo: 'skip',
                showlegend: false,
            },
        ];

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(4, 9, 18, 0.82)',
            font: { color: '#e2e8f0', family: 'Inter, sans-serif' },
            margin: { l: 48, r: 18, t: 20, b: 44 },
            hovermode: 'closest',
            legend: { orientation: 'h', y: 1.08, x: 0, bgcolor: 'rgba(0,0,0,0)' },
            xaxis: { title: 'Componente principal 1', gridcolor: 'rgba(148,163,184,0.1)', zerolinecolor: 'rgba(148,163,184,0.16)' },
            yaxis: { title: 'Componente principal 2', gridcolor: 'rgba(148,163,184,0.1)', zerolinecolor: 'rgba(148,163,184,0.16)' },
        };

        Plotly.newPlot(historySemanticView.plot, traces, layout, {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
        });

        historySemanticView.neighborSummary.textContent = `Entre los ${historySemanticState.neighborCount} vecinos más cercanos, ${neighbors.filter(point => point.label === 'phishing').length} son phishing y ${neighbors.filter(point => point.label === 'legitimate').length} son legítimos.`;
        const nearestPhishing = getPointByIndex(points, semanticMap.nearest_by_class?.phishing?.index);
        const nearestLegitimate = getPointByIndex(points, semanticMap.nearest_by_class?.legitimate?.index);
        historySemanticView.nearestPhishingMetric.innerHTML = nearestPhishing
            ? `Distancia ${formatMetric(semanticMap.nearest_by_class.phishing.distance)} | similitud ${formatMetric(semanticMap.nearest_by_class.phishing.similarity)}<br><span>${escapeHtml(nearestPhishing.subject_preview || nearestPhishing.subject || '')}</span>`
            : 'Sin datos.';
        historySemanticView.nearestLegitimateMetric.innerHTML = nearestLegitimate
            ? `Distancia ${formatMetric(semanticMap.nearest_by_class.legitimate.distance)} | similitud ${formatMetric(semanticMap.nearest_by_class.legitimate.similarity)}<br><span>${escapeHtml(nearestLegitimate.subject_preview || nearestLegitimate.subject || '')}</span>`
            : 'Sin datos.';
        historySemanticView.centroidMetrics.innerHTML = `Phishing: ${formatMetric(semanticMap.centroid_distances?.phishing)}<br>Legítimo: ${formatMetric(semanticMap.centroid_distances?.legitimate)}`;
        updateHoverCard(null, historySemanticView.hoverDetails);
        setSemanticDebug(`Historial: scatter renderizado con ${traces.length} capas visuales.`, historySemanticView.debug);

        if (typeof historySemanticView.plot.removeAllListeners === 'function') {
            historySemanticView.plot.removeAllListeners('plotly_hover');
            historySemanticView.plot.removeAllListeners('plotly_unhover');
        }
        historySemanticView.plot.on('plotly_hover', (event) => {
            const point = event?.points?.[0]?.customdata;
            if (point && (point.subject_preview || point.subject)) {
                updateHoverCard(point, historySemanticView.hoverDetails);
            }
        });
        historySemanticView.plot.on('plotly_unhover', () => updateHoverCard(null, historySemanticView.hoverDetails));
    }

    function renderResultView(target, data, options = {}) {
        const {
            playAudio = false,
            updateBackground = false,
            showActions = false,
            showResults = false,
        } = options;

        target.xaiPanel.className = 'xai-panel glass-panel';

        if (data.status === 'phishing') {
            if (playAudio) audio.playAlarm();
            target.xaiPanel.classList.add('danger-theme');
            target.threatIcon.textContent = '⚠️';
            target.threatTitle.textContent = 'Alto Riesgo: Phishing Detectado';
            if (updateBackground) {
                document.querySelector('.glow-orb.orb-1').style.background = 'rgba(239, 68, 68, 0.2)';
            }
        } else {
            if (playAudio) audio.playSafe();
            target.xaiPanel.classList.add('safe-theme');
            target.threatIcon.textContent = '🛡️';
            target.threatTitle.textContent = 'Tráfico Seguro: Correo Legítimo';
            if (updateBackground) {
                document.querySelector('.glow-orb.orb-1').style.background = 'rgba(16, 185, 129, 0.2)';
            }
        }

        target.threatConfidence.textContent = `Certeza: ${Number(data.confidence).toFixed(1)}%`;
        target.naturalExplanation.textContent = data.explanation || 'Sin diagnóstico disponible.';
        applySemanticReasoning(target, data.semantic_map);
        target.emailSubjectPrev.textContent = data.subject || 'Sin asunto';
        target.emailBodySim.textContent = data.fake_body || 'No se generó reconstrucción de cuerpo.';

        target.keywordChips.innerHTML = '';
        if (data.keywords && data.keywords.length > 0) {
            data.keywords.forEach(kw => {
                const span = document.createElement('span');
                const isPositive = kw.positive !== false;
                span.className = 'chip' + (isPositive ? ' chip-danger' : ' chip-safe');
                span.title = `${kw.word}: ${kw.impact}% | ${isPositive ? 'empuja hacia phishing' : 'empuja hacia legítimo'}`;
                const wordSpan = document.createElement('span');
                wordSpan.textContent = kw.word;
                const impactBadge = document.createElement('span');
                impactBadge.className = 'impact-badge';
                if (kw.impact > 30) impactBadge.classList.add('high-impact');
                impactBadge.textContent = `${kw.impact}%`;
                span.appendChild(wordSpan);
                span.appendChild(impactBadge);
                target.keywordChips.appendChild(span);
            });
        } else {
            target.keywordChips.innerHTML = '<span class="chip">Sin desencadenantes de palabras (señal global del embedding)</span>';
        }

        if (showResults) resultsSection.classList.remove('hidden');
        if (showActions) resultActions.classList.remove('hidden');
    }

    function displayResults(data) {
        lastResult = data;
        masterReasoningPanel.classList.add('hidden'); // panel redundante, siempre oculto
        renderResultView(
            {
                xaiPanel,
                threatIcon,
                threatTitle,
                threatConfidence,
                naturalExplanation,
                semanticReasoningText,
                keywordChips,
                emailSubjectPrev,
                emailBodySim,
            },
            data,
            { playAudio: true, updateBackground: true, showActions: true, showResults: true }
        );
        renderSemanticMap(data.semantic_map, data.semantic_error);
        if (data.analysis_id) {
            getPlotImageDataUrl(semanticPlot)
                .then((scatterPngBase64) => scatterPngBase64
                    ? persistFrontendAssets(data.analysis_id, { scatterPngBase64 })
                    : null)
                .catch((error) => console.warn('No se pudo persistir el scatter del análisis actual:', error));
        }
    }

    function applyNeighborCountFromUi() {
        if (!neighborCountInput) return;
        const nextCount = Number(neighborCountInput.value);
        if (!Number.isFinite(nextCount)) return;
        semanticState.neighborCount = nextCount;
        if (neighborCountValue) {
            neighborCountValue.textContent = String(nextCount);
        }
        setSemanticDebug(`Ajustando vecinos resaltados a ${semanticState.neighborCount}.`);
        if (semanticState.data) {
            renderSemanticMap(semanticState.data);
        }
    }

    ['input', 'change', 'mouseup', 'touchend'].forEach((eventName) => {
        neighborCountInput?.addEventListener(eventName, applyNeighborCountFromUi);
    });
    if (neighborCountInput) {
        neighborCountInput.oninput = applyNeighborCountFromUi;
        neighborCountInput.onchange = applyNeighborCountFromUi;
    }

    semanticModeBtn?.addEventListener('click', () => {
        semanticState.neighborsOnly = !semanticState.neighborsOnly;
        semanticModeBtn.textContent = semanticState.neighborsOnly ? 'Mostrar todos los puntos' : 'Mostrar solo vecinos';
        setSemanticDebug(`Cambiando modo de visualización a ${semanticState.neighborsOnly ? 'solo vecinos' : 'todos los puntos'}.`);
        if (semanticState.data) {
            renderSemanticMap(semanticState.data);
        }
    });

    async function fetchSemanticMapForResult(resultData) {
        if (resultData?.semantic_map) {
            return {
                semantic_map: resultData.semantic_map,
                semantic_error: resultData.semantic_error || null,
            };
        }

        const response = await fetch('/api/semantic-map', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                subject: resultData?.subject || '',
                status: resultData?.status || null,
            }),
        });

        if (!response.ok) {
            throw new Error('No se pudo reconstruir el mapa semántico del historial.');
        }

        return response.json();
    }

    async function openHistoryModal(data) {
        activeHistoryResult = data;
        renderResultView(historyView, data);
        historySemanticView.panel.classList.remove('hidden');
        historySemanticView.interpretation.textContent = 'Reconstruyendo mapa semántico archivado...';
        historySemanticView.note.textContent = 'El modal histórico está solicitando el scatter real del dataset para este asunto.';
        historySemanticView.plot.innerHTML = '<div class="semantic-placeholder">Reconstruyendo scatter semántico archivado...</div>';
        historySemanticView.neighborSummary.textContent = 'Reconstruyendo...';
        historySemanticView.nearestPhishingMetric.textContent = 'Reconstruyendo...';
        historySemanticView.nearestLegitimateMetric.textContent = 'Reconstruyendo...';
        historySemanticView.centroidMetrics.textContent = 'Reconstruyendo...';
        setSemanticDebug('Historial: solicitando mapa semántico al backend...', historySemanticView.debug);
        historyModalOverlay.classList.remove('hidden');
        historyModalOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';

        try {
            const semanticPayload = await fetchSemanticMapForResult(data);
            activeHistoryResult = {
                ...data,
                semantic_map: semanticPayload.semantic_map,
                semantic_error: semanticPayload.semantic_error,
            };
            applySemanticReasoning(historyView, activeHistoryResult.semantic_map);
            renderHistorySemanticMap(activeHistoryResult.semantic_map, activeHistoryResult.semantic_error);
        } catch (error) {
            setSemanticDebug(`Historial: ${error.message}`, historySemanticView.debug);
            renderHistorySemanticMap(null, error.message);
        }
    }

    function closeHistoryModal() {
        activeHistoryResult = null;
        historyModalOverlay.classList.add('hidden');
        historyModalOverlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
    }

    async function loadSvgAsPngDataUrl(svgPath, targetWidth = 256, targetHeight = 256) {
        const response = await fetch(svgPath);
        if (!response.ok) {
            throw new Error(`No se pudo cargar el logo SVG: ${svgPath}`);
        }

        const svgText = await response.text();
        const svgBlob = new Blob([svgText], { type: 'image/svg+xml;charset=utf-8' });
        const objectUrl = URL.createObjectURL(svgBlob);

        try {
            const img = await new Promise((resolve, reject) => {
                const image = new Image();
                image.onload = () => resolve(image);
                image.onerror = () => reject(new Error('No se pudo rasterizar el logo SVG.'));
                image.src = objectUrl;
            });

            const canvas = document.createElement('canvas');
            canvas.width = targetWidth;
            canvas.height = targetHeight;
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, targetWidth, targetHeight);
            ctx.drawImage(img, 0, 0, targetWidth, targetHeight);
            return canvas.toDataURL('image/png');
        } finally {
            URL.revokeObjectURL(objectUrl);
        }
    }

    async function getPlotImageDataUrl(plotElement) {
        if (!plotElement || !window.Plotly || typeof window.Plotly.toImage !== 'function') {
            return null;
        }

        try {
            const exportHost = document.createElement('div');
            exportHost.style.position = 'fixed';
            exportHost.style.left = '-10000px';
            exportHost.style.top = '0';
            exportHost.style.width = '1400px';
            exportHost.style.height = '920px';
            exportHost.style.background = '#ffffff';
            document.body.appendChild(exportHost);

            try {
                const sourceData = (plotElement.data || []).map(trace => ({ ...trace }));
                const sourceLayout = { ...(plotElement.layout || {}) };
                const exportData = sourceData.map(trace => {
                    const cloned = { ...trace };
                    if (cloned.mode === 'markers+text' && cloned.name === 'Asunto analizado') {
                        cloned.textfont = {
                            ...(cloned.textfont || {}),
                            color: '#111827',
                            size: 16,
                        };
                    }
                    return cloned;
                });

                const exportLayout = {
                    ...sourceLayout,
                    paper_bgcolor: '#ffffff',
                    plot_bgcolor: '#ffffff',
                    font: { color: '#111827', family: 'Inter, sans-serif', size: 15 },
                    legend: {
                        ...(sourceLayout.legend || {}),
                        bgcolor: 'rgba(255,255,255,0.96)',
                        bordercolor: '#cbd5e1',
                        borderwidth: 1,
                        font: { color: '#111827', size: 13 },
                    },
                    margin: { l: 90, r: 30, t: 30, b: 80 },
                    xaxis: {
                        ...(sourceLayout.xaxis || {}),
                        title: { text: 'Componente principal 1', font: { color: '#111827', size: 15 } },
                        gridcolor: '#dbe4ee',
                        zerolinecolor: '#94a3b8',
                        tickfont: { color: '#374151', size: 12 },
                    },
                    yaxis: {
                        ...(sourceLayout.yaxis || {}),
                        title: { text: 'Componente principal 2', font: { color: '#111827', size: 15 } },
                        gridcolor: '#dbe4ee',
                        zerolinecolor: '#94a3b8',
                        tickfont: { color: '#374151', size: 12 },
                    },
                };

                await window.Plotly.newPlot(exportHost, exportData, exportLayout, {
                    responsive: false,
                    displaylogo: false,
                    staticPlot: true,
                });
                await new Promise(resolve => window.requestAnimationFrame(() => resolve()));

                return await window.Plotly.toImage(exportHost, {
                    format: 'png',
                    width: 1800,
                    height: 1100,
                    scale: 2,
                });
            } finally {
                if (typeof window.Plotly.purge === 'function') {
                    window.Plotly.purge(exportHost);
                }
                exportHost.remove();
            }
        } catch (error) {
            console.warn('No se pudo rasterizar el scatter semántico para el PDF:', error);
            return null;
        }
    }

    async function persistFrontendAssets(analysisId, assets = {}) {
        if (!analysisId) return;
        const response = await fetch('/api/frontend-analysis-assets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                analysis_id: analysisId,
                scatter_png_base64: assets.scatterPngBase64 || null,
                pdf_base64: assets.pdfBase64 || null,
            }),
        });
        if (!response.ok) {
            throw new Error('No se pudieron persistir los artefactos del análisis frontend.');
        }
        return response.json();
    }

    async function exportResultToPdf(resultData, options = {}) {
        if (!resultData) return;

        const plotElement = options.plotElement || null;

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF({ unit: 'mm', format: 'a4', orientation: 'portrait' });
        const pageW = doc.internal.pageSize.getWidth();
        const pageH = doc.internal.pageSize.getHeight();
        const margin = 16;
        const contentW = pageW - (margin * 2);
        let y = 24;

        const isPhishing = resultData.status === 'phishing';
        const verdictLabel = isPhishing ? 'Phishing detectado' : 'Correo legitimo';
        const reportDate = new Date().toLocaleString('es-MX');
        const techList = runtimeTechnologies.length > 0
            ? runtimeTechnologies
            : ['Leave-One-Out XAI', 'FastAPI'];

        const C = {
            bg: [15, 17, 26],
            panel: [20, 24, 39],
            panelSoft: [31, 41, 61],
            accent: [59, 130, 246],
            accentSoft: [96, 165, 250],
            dim: [148, 163, 184],
            text: [226, 232, 240],
            danger: [239, 68, 68],
            safe: [16, 185, 129],
            white: [248, 250, 252],
        };

        function paintPageBackground() {
            doc.setFillColor(...C.bg);
            doc.rect(0, 0, pageW, pageH, 'F');
        }

        function roundedPanel(x, yPos, w, h, fill = C.panel, border = C.panelSoft, radius = 6) {
            doc.setFillColor(...fill);
            doc.setDrawColor(...border);
            doc.setLineWidth(0.25);
            doc.roundedRect(x, yPos, w, h, radius, radius, 'FD');
        }

        function guardModern(needed = 8) {
            if (y + needed > pageH - 20) {
                doc.addPage();
                paintPageBackground();
                y = 22;
            }
        }

        function sectionModern(title, eyebrow = '') {
            guardModern(18);
            if (eyebrow) {
                doc.setFontSize(7.5);
                doc.setFont('helvetica', 'bold');
                doc.setTextColor(...C.dim);
                doc.text(eyebrow.toUpperCase(), margin, y);
                y += 5;
            }
            doc.setFontSize(10.5);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(...C.accent);
            doc.text(title, margin, y);
            y += 3;
            doc.setDrawColor(...C.accentSoft);
            doc.setLineWidth(0.35);
            doc.line(margin, y, pageW - margin, y);
            y += 5;
        }

        function bodyModern(text, size = 10, bold = false, rgb = C.text, width = contentW, x = margin) {
            doc.setFontSize(size);
            doc.setFont('helvetica', bold ? 'bold' : 'normal');
            doc.setTextColor(...rgb);
            const lines = doc.splitTextToSize(String(text), width);
            const lineH = size * 0.56;
            lines.forEach((line) => {
                guardModern(lineH + 2);
                doc.text(line, x, y);
                y += lineH;
            });
            y += 2.5;
        }

        function pillRow(items, startY) {
            let x = margin;
            let currentY = startY;
            items.forEach((item) => {
                const label = String(item);
                doc.setFontSize(7.6);
                doc.setFont('helvetica', 'bold');
                const textW = doc.getTextWidth(label);
                const pillW = textW + 8;
                if (x + pillW > pageW - margin) {
                    x = margin;
                    currentY += 8;
                }
                roundedPanel(x, currentY - 4.6, pillW, 6.4, C.panelSoft, C.accentSoft, 3);
                doc.setTextColor(...C.white);
                doc.text(label, x + 4, currentY);
                x += pillW + 3;
            });
            return currentY + 8;
        }

        function drawFooterModern() {
            const total = doc.internal.getNumberOfPages();
            for (let p = 1; p <= total; p++) {
                doc.setPage(p);
                doc.setDrawColor(...C.accentSoft);
                doc.setLineWidth(0.25);
                doc.line(margin, pageH - 12, pageW - margin, pageH - 12);
                doc.setFontSize(7.5);
                doc.setFont('helvetica', 'normal');
                doc.setTextColor(...C.dim);
                doc.text('Phishing XAI Shield', margin, pageH - 7);
                doc.text(`Pag. ${p} / ${total}`, pageW - margin - 14, pageH - 7);
            }
        }

        let logoDataUrl = null;
        try {
            logoDataUrl = await loadSvgAsPngDataUrl('logo-corporate.svg', 320, 320);
        } catch (error) {
            console.warn('Fallo al cargar el logo corporativo para el PDF:', error);
        }
        const semanticNarrative = buildSemanticNarrative(resultData.semantic_map);
        const semanticPlotDataUrl = await getPlotImageDataUrl(plotElement);

        paintPageBackground();

        roundedPanel(margin, 12, contentW, 34);
        roundedPanel(margin + 4, 16, 18, 26, C.panelSoft, C.accentSoft, 5);
        if (logoDataUrl) {
            doc.addImage(logoDataUrl, 'PNG', margin + 5.5, 17.5, 15, 15);
        } else {
            doc.setFillColor(...C.accent);
            doc.roundedRect(margin + 8.5, 20.5, 9, 17, 3, 3, 'F');
        }
        doc.setFontSize(18);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(...C.white);
        doc.text('Phishing XAI Shield', margin + 28, 24);
        doc.setFontSize(8.6);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(...C.dim);
        doc.text('Forensic Threat Intelligence Report', margin + 28, 31);
        doc.text(reportDate, margin + 28, 37);
        y = 54;

        const cardGap = 4;
        const cardW = (contentW - (cardGap * 2)) / 3;
        const verdictTone = isPhishing ? C.danger : C.safe;
        [
            { title: 'Veredicto', value: verdictLabel, tone: verdictTone },
            { title: 'Certeza', value: `${Number(resultData.confidence).toFixed(1)}%`, tone: verdictTone },
            { title: 'Pipeline', value: techList[0] || 'XAI', tone: C.accentSoft },
        ].forEach((card, idx) => {
            const x = margin + (idx * (cardW + cardGap));
            roundedPanel(x, y, cardW, 22);
            doc.setFontSize(7.5);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(...C.dim);
            doc.text(card.title.toUpperCase(), x + 4, y + 6);
            doc.setFontSize(10.2);
            doc.setFont('helvetica', 'bold');
            doc.setTextColor(...card.tone);
            const lines = doc.splitTextToSize(card.value, cardW - 8);
            doc.text(lines[0], x + 4, y + 14);
        });
        y += 30;

        sectionModern('Tecnologias usadas', 'Platform');
        y = pillRow(techList, y);

        sectionModern('Sujeto analizado', 'Input');
        bodyModern(resultData.subject, 11.4, true, C.white);

        if (resultData.keywords && resultData.keywords.length > 0) {
            sectionModern('Atribucion por palabra', 'Explainability');
            const barMaxW = contentW * 0.33;
            const colWord = margin;
            const colImpact = margin + 72;
            const colBar = margin + 95;
            resultData.keywords.forEach((kw) => {
                guardModern(9);
                const isPositive = kw.positive !== false;
                const barRgb = isPositive ? C.danger : C.safe;
                const direction = isPositive ? 'empuja a phishing' : 'empuja a legitimo';
                const barW = Math.max(2, (kw.impact / 100) * barMaxW);

                doc.setFontSize(9);
                doc.setFont('helvetica', 'bold');
                doc.setTextColor(...C.text);
                const wordLines = doc.splitTextToSize(`"${kw.word}"`, 64);
                doc.text(wordLines[0], colWord, y);

                doc.setTextColor(...barRgb);
                doc.text(`${kw.impact}%`, colImpact, y);

                doc.setFillColor(...C.panelSoft);
                doc.roundedRect(colBar, y - 3.8, barMaxW, 4.4, 2, 2, 'F');
                doc.setFillColor(...barRgb);
                doc.roundedRect(colBar, y - 3.8, barW, 4.4, 2, 2, 'F');

                doc.setFontSize(7.1);
                doc.setFont('helvetica', 'normal');
                doc.setTextColor(...C.dim);
                doc.text(direction, colBar + barMaxW + 4, y);
                y += 8;
            });
            y += 1.5;
        }

        if (resultData.explanation) {
            sectionModern('Razonamiento maestro', 'AI Narrative');
            bodyModern(resultData.explanation, 10.2, false, C.text);
        }

        if (semanticNarrative || resultData.semantic_map) {
            sectionModern('Lectura del mapa semántico', 'Semantic Map');
            if (semanticNarrative) {
                bodyModern(semanticNarrative, 10.2, false, C.text);
            }
            if (resultData.semantic_map) {
                const nearestPhishingDistance = formatMetric(resultData.semantic_map?.nearest_by_class?.phishing?.distance);
                const nearestLegitimateDistance = formatMetric(resultData.semantic_map?.nearest_by_class?.legitimate?.distance);
                const centroidPhishingDistance = formatMetric(resultData.semantic_map?.centroid_distances?.phishing);
                const centroidLegitimateDistance = formatMetric(resultData.semantic_map?.centroid_distances?.legitimate);
                bodyModern(
                    `Punto phishing más cercano: ${nearestPhishingDistance} | Punto legítimo más cercano: ${nearestLegitimateDistance} | Centroide phishing: ${centroidPhishingDistance} | Centroide legítimo: ${centroidLegitimateDistance}`,
                    9.2,
                    false,
                    C.dim
                );
            }
            if (semanticPlotDataUrl) {
                const imageProps = doc.getImageProperties(semanticPlotDataUrl);
                const maxImageWidth = contentW - 8;
                const maxImageHeight = 110;
                const imageAspect = imageProps.width / imageProps.height;

                let renderWidth = maxImageWidth;
                let renderHeight = renderWidth / imageAspect;
                if (renderHeight > maxImageHeight) {
                    renderHeight = maxImageHeight;
                    renderWidth = renderHeight * imageAspect;
                }

                const panelHeight = renderHeight + 8;
                guardModern(panelHeight + 8);
                roundedPanel(margin, y, contentW, panelHeight, C.panel, C.panelSoft, 6);
                const imageX = margin + ((contentW - renderWidth) / 2);
                doc.addImage(semanticPlotDataUrl, 'PNG', imageX, y + 4, renderWidth, renderHeight);
                y += panelHeight + 6;
            }
        }

        const hasBody = resultData.fake_body &&
                        resultData.fake_body !== 'No se genero reconstruccion de cuerpo.' &&
                        resultData.fake_body !== 'No se gener? reconstrucci?n de cuerpo.';
        if (hasBody) {
            sectionModern(
                isPhishing ? 'Reconstruccion del vector de ataque' : 'Cuerpo de mensaje reconstruido',
                'Message Body'
            );
            bodyModern(resultData.fake_body, 10, false, C.text);
        }

        drawFooterModern();
        if (resultData.analysis_id) {
            try {
                await persistFrontendAssets(resultData.analysis_id, {
                    pdfBase64: doc.output('datauristring'),
                });
            } catch (error) {
                console.warn('No se pudo persistir el PDF del análisis frontend:', error);
            }
        }
        doc.save(`Reporte_Forense_XAI_${Date.now()}.pdf`);
    }

    // --- Export Logic ---
    exportPdfBtn.addEventListener('click', async () => {
        await exportResultToPdf(lastResult, { plotElement: semanticPlot });
    });

    exportHistoryPdfBtn.addEventListener('click', async () => {
        await exportResultToPdf(activeHistoryResult, { plotElement: historySemanticView.plot });
    });

    // --- History Logic ---
    openHistoryBtn.addEventListener('click', () => {
        historySidebar.classList.add('open');
        loadHistory();
    });

    closeHistoryBtn.addEventListener('click', () => {
        historySidebar.classList.remove('open');
    });

    closeHistoryModalBtn.addEventListener('click', () => {
        closeHistoryModal();
    });

    historyModalOverlay.addEventListener('click', (event) => {
        if (event.target === historyModalOverlay) {
            closeHistoryModal();
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && !historyModalOverlay.classList.contains('hidden')) {
            closeHistoryModal();
        }
    });

    async function loadHistory() {
        try {
            const resp = await fetch('/api/history');
            const data = await resp.json();
            renderHistory(data);
        } catch (err) { console.error("Error cargando historial:", err); }
    }

    function renderHistory(items) {
        historyList.innerHTML = '';
        if (items.length === 0) {
            historyList.innerHTML = '<p class="empty-msg">No hay análisis previos.</p>';
            return;
        }
        items.forEach(item => {
            const card = document.createElement('div');
            card.className = `history-card ${item.status}`;
            card.innerHTML = `
                <div class="card-header">
                    <span class="card-status">${item.status}</span>
                    <span class="card-time">${formatTime(item.timestamp)}</span>
                </div>
                <div class="card-subject">${item.subject}</div>
            `;
            card.addEventListener('click', () => {
                openHistoryModal(item);
                historySidebar.classList.remove('open');
            });
            historyList.appendChild(card);
        });
    }

    function formatTime(isoStr) {
        try {
            const date = new Date(isoStr);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch { return isoStr; }
    }

    loadHistory();
});
