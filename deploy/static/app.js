const configNode = document.getElementById("runtimeConfig");
const resultCard = document.getElementById("resultCard");
const verdictTitle = document.getElementById("verdictTitle");
const confidenceBadge = document.getElementById("confidenceBadge");
const explanationText = document.getElementById("explanationText");
const bodyText = document.getElementById("bodyText");
const keywordChips = document.getElementById("keywordChips");
const analyzeBtn = document.getElementById("analyzeBtn");
const subjectInput = document.getElementById("subjectInput");

async function loadConfig() {
    try {
        const response = await fetch("/api/config");
        const config = await response.json();
        configNode.textContent = `Model: ${config.embedding} + ${config.classifier} | Run: ${config.run_id}`;
    } catch (error) {
        configNode.textContent = "Unable to load deploy configuration.";
    }
}

function renderKeywords(keywords) {
    keywordChips.innerHTML = "";
    if (!keywords || keywords.length === 0) {
        keywordChips.innerHTML = '<span class="chip neutral">No dominant keywords isolated</span>';
        return;
    }

    keywords.forEach((item) => {
        const chip = document.createElement("span");
        chip.className = `chip ${item.positive !== false ? "danger" : "safe"}`;
        chip.textContent = `${item.word} (${item.impact}%)`;
        keywordChips.appendChild(chip);
    });
}

async function analyze() {
    const subject = subjectInput.value.trim();
    if (!subject) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ subject }),
        });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Analysis failed");
        }

        resultCard.classList.remove("hidden");
        verdictTitle.textContent = data.status === "phishing" ? "Phishing detected" : "Legitimate message";
        confidenceBadge.textContent = `${Number(data.confidence).toFixed(1)}%`;
        confidenceBadge.className = `badge ${data.status === "phishing" ? "danger" : "safe"}`;
        explanationText.textContent = data.explanation || "No explanation available.";
        bodyText.textContent = data.fake_body || "No synthetic body available.";
        renderKeywords(data.keywords || []);
    } catch (error) {
        alert(error.message);
    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Analyze";
    }
}

analyzeBtn.addEventListener("click", analyze);
subjectInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
        analyze();
    }
});

loadConfig();
