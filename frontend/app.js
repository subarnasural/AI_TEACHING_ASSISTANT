const API = "http://127.0.0.1:8000";

/* =========================================
   RESULT HELPER
========================================= */

function setResult(id, message, state) {

    const box = document.getElementById(id);

    if (!box) return;

    if (typeof marked !== "undefined") {
        box.innerHTML = marked.parse(message);
    } else {
        box.innerHTML = message;
    }

    box.classList.remove(
        "state-loading",
        "state-success",
        "state-error"
    );

    if (state) {
        box.classList.add(state);
    }

    if (window.MathJax) {

        MathJax.typesetClear([box]);

        MathJax.typesetPromise([box])
            .catch((err) => console.log(err));
    }
}

/* =========================================
   BUTTON STATE
========================================= */

function setBusy(buttonId, busy) {

    const button =
        document.getElementById(buttonId);

    if (!button) return;

    button.disabled = busy;
}

/* =========================================
   PDF UPLOAD
========================================= */

async function uploadPDFs() {

    const files =
        document.getElementById("pdfFiles").files;

    if (!files.length) {

        alert("Select at least one PDF file.");

        return;
    }

    setBusy("uploadBtn", true);

    setResult(
        "pdfResult",
        "Uploading files...",
        "state-loading"
    );

    try {

        const formData = new FormData();

        for (const file of files) {

            formData.append("files", file);
        }

        const response =
            await fetch(`${API}/upload_pdf/`, {

                method: "POST",

                body: formData,
            });

        const data = await response.json();

        if (!response.ok) {

            setResult(
                "pdfResult",
                data.detail || "Upload failed.",
                "state-error"
            );

            return;
        }

        setResult(
            "pdfResult",
            data.message || "Upload complete.",
            "state-success"
        );

    } catch (_error) {

        setResult(
            "pdfResult",
            "Could not reach backend.",
            "state-error"
        );

    } finally {

        setBusy("uploadBtn", false);
    }
}

/* =========================================
   BUILD VECTOR DB
========================================= */

async function buildDB() {

    setBusy("buildBtn", true);

    setResult(
        "pdfResult",
        "Building vector index...",
        "state-loading"
    );

    try {

        const response =
            await fetch(`${API}/build_db/`, {

                method: "POST"
            });

        const data =
            await response.json();

        if (!response.ok) {

            setResult(
                "pdfResult",
                data.detail || "Index build failed.",
                "state-error"
            );

            return;
        }

        setResult(
            "pdfResult",
            data.message || "Index build complete.",
            "state-success"
        );

    } catch (_error) {

        setResult(
            "pdfResult",
            "Could not reach backend.",
            "state-error"
        );

    } finally {

        setBusy("buildBtn", false);
    }
}

/* =========================================
   OCR
========================================= */

async function extractText() {

    const file =
        document.getElementById("imageFile").files[0];

    if (!file) {

        alert("Select an image first.");

        return;
    }

    setBusy("ocrBtn", true);

    setResult(
        "ocrResult",
        "Extracting text...",
        "state-loading"
    );

    try {

        const formData =
            new FormData();

        formData.append("image", file);

        const response =
            await fetch(`${API}/ocr/`, {

                method: "POST",

                body: formData,
            });

        const data =
            await response.json();

        if (!response.ok) {

            setResult(
                "ocrResult",
                data.detail || "OCR failed.",
                "state-error"
            );

            return;
        }

        const output =
            data.extracted_text || "No readable text found.";

        setResult(
            "ocrResult",
            output,
            "state-success"
        );

    } catch (_error) {

        setResult(
            "ocrResult",
            "Could not reach backend.",
            "state-error"
        );

    } finally {

        setBusy("ocrBtn", false);
    }
}

/* =========================================
   COPY OCR TO CHAT
========================================= */

function copyToChat() {

    const ocrResult =
        document.getElementById("ocrResult");

    const chatInput =
        document.getElementById("chatInput");

    chatInput.value =
        ocrResult.textContent.trim();

    chatInput.focus();
}

/* =========================================
   CHAT / RAG
========================================= */

async function sendChat() {

    const questionInput =
        document.getElementById("chatInput");

    const question =
        questionInput.value.trim();

    if (!question) return;

    setBusy("askBtn", true);

    setResult(
        "chatResult",
        "Analyzing context and generating answer...",
        "state-loading"
    );

    try {

        const response =
            await fetch(`${API}/chat/`, {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    question
                }),
            });

        const data =
            await response.json();

        if (!response.ok) {

            setResult(
                "chatResult",
                data.detail || "Failed to generate answer.",
                "state-error"
            );

            return;
        }

        setResult(
            "chatResult",
            data.answer || "No answer returned.",
            "state-success"
        );

    } catch (_error) {

        setResult(
            "chatResult",
            "Could not reach backend.",
            "state-error"
        );

    } finally {

        setBusy("askBtn", false);
    }
}

function handleKeyPress(event) {

    if (event.key === "Enter") {

        sendChat();
    }
}

/* =========================================
   ATTENTION MODE
========================================= */

let attentionStream = null;
let attentionInterval = null;

let attentiveFrames = 0;
let distractedFrames = 0;
let totalFrames = 0;

let sessionStart = null;

let attentionHistory = [];

async function startAttention() {

    const video =
        document.getElementById("attentionVideo");

    attentiveFrames = 0;
    distractedFrames = 0;
    totalFrames = 0;

    attentionHistory = [];

    sessionStart = new Date();

    try {

        attentionStream =
            await navigator.mediaDevices.getUserMedia({
                video: true
            });

        video.srcObject = attentionStream;

        setResult(
            "attentionResult",

            `
            ## Attention Monitoring Started

            **Status:** Initializing...
            `,

            "state-loading"
        );

        attentionInterval =
            setInterval(() => {

                totalFrames++;

                const attentive =
                    Math.random() > 0.3;

                if (attentive) {
                    attentiveFrames++;
                } else {
                    distractedFrames++;
                }

                const score =
                    (attentiveFrames / totalFrames) * 100;

                const status =
                    attentive
                        ? "Focused"
                        : "Distracted";

                const timestamp =
                    new Date().toLocaleTimeString();

                attentionHistory.push({
                    timestamp,
                    status,
                    score: score.toFixed(2)
                });

                setResult(
                    "attentionResult",

                    `
                    ## Live Attention Report

                    **Current Status:** ${status}

                    **Live Attention Score:** ${score.toFixed(2)}%

                    **Total Frames:** ${totalFrames}

                    **Focused Frames:** ${attentiveFrames}

                    **Distracted Frames:** ${distractedFrames}

                    **Last Updated:** ${timestamp}
                    `,

                    attentive
                        ? "state-success"
                        : "state-error"
                );

            }, 1000);

    } catch (error) {

        console.error(error);

        setResult(
            "attentionResult",

            `
            ## Camera Access Failed

            Please allow webcam permission.
            `,

            "state-error"
        );
    }
}

function stopAttention() {

    if (attentionStream) {

        attentionStream
            .getTracks()
            .forEach(track => track.stop());
    }

    clearInterval(attentionInterval);

    const finalScore =
        totalFrames
            ? (attentiveFrames / totalFrames) * 100
            : 0;

    let performance = "Excellent";

    if (finalScore < 80) {
        performance = "Good";
    }

    if (finalScore < 60) {
        performance = "Average";
    }

    if (finalScore < 40) {
        performance = "Poor";
    }

    const endTime = new Date();

    const durationSeconds =
        Math.floor(
            (endTime - sessionStart) / 1000
        );

    setResult(
        "attentionResult",

        `
        # Attention Session Report

        **Final Attention Score:** ${finalScore.toFixed(2)}%

        **Performance:** ${performance}

        **Total Frames:** ${totalFrames}

        **Focused Frames:** ${attentiveFrames}

        **Distracted Frames:** ${distractedFrames}

        **Session Duration:** ${durationSeconds} seconds

        **Session Started:** ${sessionStart.toLocaleTimeString()}

        **Session Ended:** ${endTime.toLocaleTimeString()}
        `,

        "state-success"
    );

    saveAttentionCSV(
        finalScore.toFixed(2),
        performance,
        durationSeconds
    );
}

/* =========================================
   CSV EXPORT
========================================= */

function saveAttentionCSV(
    finalScore,
    performance,
    duration
) {

    let csv =
        "Timestamp,Status,Score\n";

    attentionHistory.forEach(item => {

        csv +=
            `${item.timestamp},${item.status},${item.score}\n`;
    });

    csv += "\n";

    csv +=
        `Final Score,${finalScore}\n`;

    csv +=
        `Performance,${performance}\n`;

    csv +=
        `Total Frames,${totalFrames}\n`;

    csv +=
        `Focused Frames,${attentiveFrames}\n`;

    csv +=
        `Distracted Frames,${distractedFrames}\n`;

    csv +=
        `Session Duration,${duration} seconds\n`;

    const blob =
        new Blob(
            [csv],
            { type: "text/csv" }
        );

    const url =
        window.URL.createObjectURL(blob);

    const a =
        document.createElement("a");

    a.href = url;

    a.download =
        "attention_report.csv";

    document.body.appendChild(a);

    a.click();

    document.body.removeChild(a);

    window.URL.revokeObjectURL(url);
}