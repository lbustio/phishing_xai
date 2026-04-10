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
    const keywordChips = document.getElementById('keywordChips');
    const emailSubjectPrev = document.getElementById('emailSubjectPrev');
    const emailBodySim = document.getElementById('emailBodySim');

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
        keywordChips: document.getElementById('historyKeywordChips'),
        emailSubjectPrev: document.getElementById('historyEmailSubjectPrev'),
        emailBodySim: document.getElementById('historyEmailBodySim'),
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

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const payload = JSON.parse(line);
                        if (payload.type === 'log') {
                            addLogEntry(payload.content);
                        } else if (payload.type === 'result') {
                            resultData = payload.data;
                        }
                    } catch (e) { console.error("Error parsing stream chunk:", e); }
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
        target.emailSubjectPrev.textContent = data.subject || 'Sin asunto';
        target.emailBodySim.textContent = data.fake_body || 'No se generó reconstrucción de cuerpo.';

        target.keywordChips.innerHTML = '';
        if (data.keywords && data.keywords.length > 0) {
            data.keywords.forEach(kw => {
                const span = document.createElement('span');
                const isPositive = kw.positive !== false;
                span.className = 'chip' + (isPositive ? ' chip-danger' : ' chip-safe');
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
                keywordChips,
                emailSubjectPrev,
                emailBodySim,
            },
            data,
            { playAudio: true, updateBackground: true, showActions: true, showResults: true }
        );
    }

    function openHistoryModal(data) {
        activeHistoryResult = data;
        renderResultView(historyView, data);
        historyModalOverlay.classList.remove('hidden');
        historyModalOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
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

    async function exportResultToPdf(resultData) {
        if (!resultData) return;

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
        doc.save(`Reporte_Forense_XAI_${Date.now()}.pdf`);
    }

    // --- Export Logic ---
    exportPdfBtn.addEventListener('click', async () => {
        await exportResultToPdf(lastResult);
    });

    exportHistoryPdfBtn.addEventListener('click', async () => {
        await exportResultToPdf(activeHistoryResult);
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
