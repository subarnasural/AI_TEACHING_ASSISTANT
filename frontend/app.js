const API = window.location.origin;

function setResult(id, message, state) {
    const box = document.getElementById(id);
    box.textContent = message;
    box.classList.remove("state-loading", "state-success", "state-error");
    if (state) {
        box.classList.add(state);
    }
}

function setBusy(buttonId, busy) {
    const button = document.getElementById(buttonId);
    if (!button) return;
    button.disabled = busy;
}

async function uploadPDFs() {
    const files = document.getElementById("pdfFiles").files;
    if (!files.length) {
        alert("Select at least one PDF file.");
        return;
    }

    setBusy("uploadBtn", true);
    setResult("pdfResult", "Uploading files...", "state-loading");

    try {
        const formData = new FormData();
        for (const file of files) {
            formData.append("files", file);
        }

        const response = await fetch(`${API}/upload_pdf/`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            setResult("pdfResult", data.detail || "Upload failed.", "state-error");
            return;
        }

        setResult("pdfResult", data.message || "Upload complete.", "state-success");
    } catch (_error) {
        setResult("pdfResult", "Could not reach the backend service.", "state-error");
    } finally {
        setBusy("uploadBtn", false);
    }
}

async function buildDB() {
    setBusy("buildBtn", true);
    setResult("pdfResult", "Building vector index. This can take a minute for large files.", "state-loading");

    try {
        const response = await fetch(`${API}/build_db/`, { method: "POST" });
        const data = await response.json();

        if (!response.ok) {
            setResult("pdfResult", data.detail || "Index build failed.", "state-error");
            return;
        }

        setResult("pdfResult", data.message || "Index build complete.", "state-success");
    } catch (_error) {
        setResult("pdfResult", "Could not reach the backend service.", "state-error");
    } finally {
        setBusy("buildBtn", false);
    }
}

async function extractText() {
    const file = document.getElementById("imageFile").files[0];
    if (!file) {
        alert("Select an image first.");
        return;
    }

    setBusy("ocrBtn", true);
    setResult("ocrResult", "Extracting text from image...", "state-loading");

    try {
        const formData = new FormData();
        formData.append("image", file);

        const response = await fetch(`${API}/ocr/`, {
            method: "POST",
            body: formData,
        });
        const data = await response.json();

        if (!response.ok) {
            setResult("ocrResult", data.detail || "OCR extraction failed.", "state-error");
            return;
        }

        const text = typeof data.extracted_text === "string"
            ? data.extracted_text
            : "No readable text was found.";
        const confidence = Number.isFinite(Number(data.confidence)) ? Number(data.confidence) : null;
        const method = typeof data.method === "string" ? data.method : null;

        const meta = [];
        if (confidence !== null) {
            meta.push(`Confidence: ${confidence.toFixed(2)}`);
        }
        if (method) {
            meta.push(`Method: ${method}`);
        }

        const output = meta.length ? `${text}\n\n${meta.join(" | ")}` : text;
        setResult("ocrResult", output, "state-success");
    } catch (_error) {
        setResult("ocrResult", "Could not reach the backend service.", "state-error");
    } finally {
        setBusy("ocrBtn", false);
    }
}

function copyToChat() {
    const ocrResult = document.getElementById("ocrResult");
    const chatInput = document.getElementById("chatInput");

    if (!ocrResult.classList.contains("state-success")) {
        alert("Extract text first.");
        return;
    }

    chatInput.value = ocrResult.textContent.trim();
    chatInput.focus();
}

async function sendChat() {
    const questionInput = document.getElementById("chatInput");
    const question = questionInput.value.trim();
    if (!question) return;

    setBusy("askBtn", true);
    setResult("chatResult", "Analyzing context and preparing answer...", "state-loading");

    try {
        const response = await fetch(`${API}/chat/`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
        });
        const data = await response.json();

        if (!response.ok) {
            setResult("chatResult", data.detail || "Failed to generate response.", "state-error");
            return;
        }

        setResult("chatResult", data.answer || "No response returned.", "state-success");
    } catch (_error) {
        setResult("chatResult", "Could not reach the backend service.", "state-error");
    } finally {
        setBusy("askBtn", false);
    }
}

function handleKeyPress(event) {
    if (event.key === "Enter") {
        sendChat();
    }
}
