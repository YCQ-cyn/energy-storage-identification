const elements = {
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    selectedFile: document.getElementById("selectedFile"),
    fileName: document.getElementById("fileName"),
    fileSize: document.getElementById("fileSize"),
    removeFile: document.getElementById("removeFile"),
    analyzeButton: document.getElementById("analyzeButton"),
    statusMessage: document.getElementById("statusMessage"),
    formatLine: document.getElementById("formatLine"),
    resultBody: document.getElementById("resultBody"),
    emptyState: document.getElementById("emptyState"),
    resultSummary: document.getElementById("resultSummary"),
    searchBox: document.getElementById("searchBox"),
    searchInput: document.getElementById("searchInput"),
    tableWrap: document.getElementById("tableWrap"),
    emptyStateTitle: document.getElementById("emptyStateTitle"),
    emptyStateMessage: document.getElementById("emptyStateMessage"),
    downloadButton: document.getElementById("downloadButton")
};

let selectedFile = null;
let resultRows = [];

const timeColumns = Array.from({length: 96}, (_, index) => {
    const minutes = index * 15;
    return `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, "0")}`;
});
elements.formatLine.textContent = ["数据日期", "户号", ...timeColumns].join("　");

elements.dropZone.addEventListener("click", () => elements.fileInput.click());
elements.fileInput.addEventListener("change", event => {
    chooseFile(event.target.files[0]);
});

["dragenter", "dragover"].forEach(type => {
    elements.dropZone.addEventListener(type, event => {
        event.preventDefault();
        elements.dropZone.classList.add("dragging");
    });
});

["dragleave", "drop"].forEach(type => {
    elements.dropZone.addEventListener(type, event => {
        event.preventDefault();
        elements.dropZone.classList.remove("dragging");
    });
});

elements.dropZone.addEventListener("drop", event => {
    chooseFile(event.dataTransfer.files[0]);
});

elements.removeFile.addEventListener("click", event => {
    event.stopPropagation();
    selectedFile = null;
    elements.fileInput.value = "";
    elements.selectedFile.hidden = true;
    elements.dropZone.hidden = false;
    elements.analyzeButton.disabled = true;
    setStatus("请选择待排查文件", "normal");
});

elements.analyzeButton.addEventListener("click", analyze);
elements.searchBox.addEventListener("submit", event => {
    event.preventDefault();
    searchRows();
});

function chooseFile(file) {
    if (!file) {
        return;
    }
    const extension = file.name.split(".").pop().toLowerCase();
    if (!["xlsx", "csv"].includes(extension)) {
        setStatus("请选择 xlsx 或 csv 文件；xls请先另存为xlsx", "error");
        return;
    }
    if (file.size > 512 * 1024 * 1024) {
        setStatus("上传文件不能超过512MB", "error");
        return;
    }
    selectedFile = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.selectedFile.hidden = false;
    elements.dropZone.hidden = true;
    elements.analyzeButton.disabled = false;
    setStatus("文件已就绪，可以开始排查", "success");
}

async function analyze() {
    if (!selectedFile) {
        return;
    }
    const policy = document.querySelector("input[name='policy']:checked").value;
    setLoading(true);
    setStatus("正在读取数据并运行储能识别模型，请勿关闭页面…", "normal");
    clearResults();
    try {
        const endpoint = `/api/analyze?policy=${encodeURIComponent(policy)}` +
            `&filename=${encodeURIComponent(selectedFile.name)}`;
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {"Content-Type": "application/octet-stream"},
            body: selectedFile
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.error || "分析失败，请检查文件格式");
        }
        resultRows = Array.isArray(payload.rows) ? payload.rows : [];
        renderRows(resultRows);
        elements.downloadButton.href = payload.downloadUrl;
        elements.downloadButton.download = payload.fileName;
        elements.downloadButton.hidden = false;
        elements.searchBox.hidden = resultRows.length === 0;
        elements.resultSummary.textContent = `共完成 ${resultRows.length} 户用户排查，可搜索预览或下载完整Excel。`;
        setStatus(`排查完成，共生成 ${resultRows.length} 户结果`, "success");
    } catch (error) {
        const message = error instanceof Error ? error.message : "分析失败";
        setStatus(message, "error");
        elements.resultSummary.textContent = "未生成结果，请根据提示检查文件后重试。";
        elements.emptyState.hidden = false;
    } finally {
        setLoading(false);
    }
}

function renderRows(rows, keyword = "") {
    elements.resultBody.replaceChildren();
    elements.emptyState.hidden = rows.length > 0;
    rows.forEach(row => {
        const tableRow = document.createElement("tr");
        tableRow.appendChild(cell(row["户号"], keyword));

        const levelCell = document.createElement("td");
        const badge = document.createElement("span");
        badge.className = `level-badge ${levelClass(row["储能情况"])}`;
        badge.textContent = row["储能情况"] ?? "-";
        levelCell.appendChild(badge);
        tableRow.appendChild(levelCell);

        tableRow.appendChild(cell(row["总天数"]));
        tableRow.appendChild(cell(row["有效天数"]));
        elements.resultBody.appendChild(tableRow);
    });
}

function searchRows() {
    const keyword = elements.searchInput.value.trim().toLowerCase();
    const matchedRows = keyword
        ? resultRows.filter(row => String(row["户号"] ?? "")
            .toLowerCase()
            .includes(keyword))
        : resultRows;

    renderRows(matchedRows, keyword);
    elements.tableWrap.scrollTop = 0;
    if (!keyword) {
        elements.resultSummary.textContent =
            `共完成 ${resultRows.length} 户用户排查，可搜索预览或下载完整Excel。`;
        return;
    }

    elements.resultSummary.textContent =
        `户号包含“${elements.searchInput.value.trim()}”的用户共 ${matchedRows.length} 户。`;
    elements.emptyStateTitle.textContent = "未找到匹配用户";
    elements.emptyStateMessage.textContent = "请修改户号关键词后重新搜索";
}

function cell(value, keyword = "") {
    const element = document.createElement("td");
    const text = String(value ?? "-");
    if (!keyword) {
        element.textContent = text;
        return element;
    }

    const normalizedText = text.toLowerCase();
    let start = 0;
    let matchIndex = normalizedText.indexOf(keyword, start);
    while (matchIndex >= 0) {
        element.append(document.createTextNode(text.slice(start, matchIndex)));
        const marker = document.createElement("mark");
        marker.className = "search-match";
        marker.textContent = text.slice(matchIndex, matchIndex + keyword.length);
        element.append(marker);
        start = matchIndex + keyword.length;
        matchIndex = normalizedText.indexOf(keyword, start);
    }
    element.append(document.createTextNode(text.slice(start)));
    return element;
}

function levelClass(level) {
    if (level === "强储能") return "level-strong";
    if (level === "中储能") return "level-medium";
    if (level === "无有效数据") return "level-empty";
    return "level-low";
}

function clearResults() {
    resultRows = [];
    elements.resultBody.replaceChildren();
    elements.emptyState.hidden = true;
    elements.downloadButton.hidden = true;
    elements.searchBox.hidden = true;
    elements.searchInput.value = "";
    elements.emptyStateTitle.textContent = "暂无排查结果";
    elements.emptyStateMessage.textContent = "上传符合建议格式的文件并开始分析";
    elements.resultSummary.textContent = "正在分析，请稍候…";
}

function setLoading(loading) {
    elements.analyzeButton.disabled = loading || !selectedFile;
    elements.analyzeButton.classList.toggle("loading", loading);
    elements.analyzeButton.querySelector(".button-text").textContent =
        loading ? "正在智能排查" : "开始智能排查";
}

function setStatus(message, type) {
    elements.statusMessage.className = "status-message";
    if (type === "success" || type === "error") {
        elements.statusMessage.classList.add(type);
    }
    elements.statusMessage.lastChild.textContent = ` ${message}`;
}

function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
