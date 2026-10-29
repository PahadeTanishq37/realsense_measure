/* ================================================================
   RealSense 3D Scanner — GUI Client JS
   Handles SocketIO communication, config building, UI state
   ================================================================ */

(function () {
  "use strict";

  // ---- Socket.IO ----
  const socket = io({ transports: ["websocket", "polling"] });

  // ---- DOM refs ----
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const els = {
    console:         $("#console"),
    statusPill:      $("#statusPill"),
    statusText:      $(".status-text"),
    elapsedTimer:    $("#elapsedTimer"),
    btnStart:        $("#btnStart"),
    btnStop:         $("#btnStop"),
    btnClear:        $("#btnClear"),
    commandText:     $("#commandText"),
    resultsBar:      $("#resultsBar"),
    resultExitCode:  $("#resultExitCode"),
    resultDuration:  $("#resultDuration"),
    resultStatus:    $("#resultStatus"),
    autoScroll:      $("#cfgAutoScroll"),
    btnScrollBottom: $("#btnScrollBottom"),
    lightbox:        $("#lightbox"),
    lightboxImg:     $("#lightboxImg"),
    lightboxClose:   $("#lightboxClose"),
    thumbnailGrid:   $("#thumbnailGrid"),
    previewCapture:  $("#previewCapture"),
    scanHistoryList: $("#scanHistoryList"),
    stageIndicators: $("#stageIndicators"),
  };

  // ---- State ----
  let isRunning = false;
  let timerInterval = null;
  let startTime = null;
  let currentStage = 0;
  let welcomeCleared = false;

  // ================================================================
  // Config → Command builder
  // ================================================================

  function getConfig() {
    const targetEl = $("#cfgTarget");
    const saveRawEl = $("#cfgSaveRaw");
    return {
      target: targetEl ? targetEl.value : "head",
      save_raw: saveRawEl ? saveRawEl.value.trim() : "",
    };
  }

  function buildCommandPreview() {
    if (!els.commandText) return;
    const cfg = getConfig();
    const saveFolder = cfg.save_raw || "raw_capture_05";
    const cmd = `python main.py --target ${cfg.target} --face-quality --gate-mode advisory --tsdf --multiway --loop-closure --save-raw ${saveFolder}`;
    els.commandText.textContent = cmd;
  }

  // ================================================================
  // Presets & Target Switching
  // ================================================================

  function syncPresetButtons(target) {
    $$(".preset-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.preset === target);
    });
  }

  function applyPreset(name) {
    const targetSelect = $("#cfgTarget");
    if (!targetSelect) return;
    targetSelect.value = (name === "body") ? "body" : "head";
    syncPresetButtons(targetSelect.value);
    buildCommandPreview();
  }

  function fetchNextRawCapture() {
    fetch("/api/next-raw-capture")
      .then((r) => r.json())
      .then((data) => {
        if (data.next_name) {
          const el = $("#cfgSaveRaw");
          if (el) {
            el.value = data.next_name;
            buildCommandPreview();
          }
        }
      })
      .catch(() => {});
  }

  // ================================================================
  // Console output
  // ================================================================

  function clearWelcome() {
    if (!welcomeCleared) {
      const w = els.console.querySelector(".console-welcome");
      if (w) w.remove();
      welcomeCleared = true;
    }
  }

  function appendLog(text, type = "info") {
    clearWelcome();
    const line = document.createElement("div");
    line.className = `log-line type-${type}`;
    line.textContent = text;
    els.console.appendChild(line);

    if (els.autoScroll.checked) {
      els.console.scrollTop = els.console.scrollHeight;
    }

    // Cap lines
    while (els.console.children.length > 2000) {
      els.console.removeChild(els.console.firstChild);
    }
  }

  // ================================================================
  // Timer
  // ================================================================

  function formatTime(secs) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  }

  function startTimer() {
    startTime = Date.now();
    timerInterval = setInterval(() => {
      const elapsed = (Date.now() - startTime) / 1000;
      els.elapsedTimer.textContent = formatTime(elapsed);
    }, 500);
  }

  function stopTimer() {
    if (timerInterval) {
      clearInterval(timerInterval);
      timerInterval = null;
    }
  }

  // ================================================================
  // Stage indicators
  // ================================================================

  function updateStageIndicators(stage) {
    $$(".stage-dot").forEach((dot) => {
      const s = parseInt(dot.dataset.stage);
      dot.classList.remove("active", "done");
      if (s < stage) dot.classList.add("done");
      else if (s === stage) dot.classList.add("active");
    });

    // Also update connectors
    $$(".stage-connector").forEach((conn, i) => {
      if (i + 1 < stage) conn.style.background = "var(--success)";
      else conn.style.background = "var(--glass-border)";
    });
  }

  // ================================================================
  // UI state management
  // ================================================================

  function setRunning(running) {
    isRunning = running;
    els.btnStart.disabled = running;
    els.btnStop.disabled = !running;

    if (running) {
      els.statusPill.className = "status-pill running";
      els.statusText.textContent = "Scanning...";
      els.resultsBar.style.display = "none";
      startTimer();
    } else {
      els.statusPill.className = "status-pill";
      els.statusText.textContent = "Ready";
      stopTimer();
    }
  }

  // ================================================================
  // SocketIO events
  // ================================================================

  socket.on("connect", () => {
    console.log("[GUI] Connected to server");
  });

  socket.on("disconnect", () => {
    console.log("[GUI] Disconnected from server");
    if (isRunning) {
      appendLog("[GUI] Connection lost to server.", "error");
      setRunning(false);
    }
  });

  socket.on("log_line", (data) => {
    appendLog(data.text, data.type || "info");
    if (data.stage && data.stage !== currentStage) {
      currentStage = data.stage;
      updateStageIndicators(currentStage);
    }
  });

  socket.on("stage_update", (data) => {
    currentStage = data.stage;
    updateStageIndicators(currentStage);
  });

  socket.on("scan_started", (data) => {
    setRunning(true);
    appendLog(`[GUI] Process started (PID: ${data.pid})`, "success");
  });

  socket.on("command_echo", (data) => {
    appendLog(`$ ${data.command}`, "stage");
  });

  socket.on("scan_finished", (data) => {
    setRunning(false);
    const elapsed = data.elapsed || 0;
    const code = data.exit_code;

    if (code === 0) {
      appendLog(`\n✓ Scan completed successfully in ${formatTime(elapsed)}.`, "success");
      els.statusPill.className = "status-pill";
      els.statusText.textContent = "Complete";
    } else if (code === -15 || code === 1) {
      appendLog(`\n⚠ Scan stopped (exit code: ${code}).`, "warning");
      els.statusPill.className = "status-pill";
      els.statusText.textContent = "Stopped";
    } else {
      appendLog(`\n✗ Scan failed (exit code: ${code}).`, "error");
      els.statusPill.className = "status-pill error";
      els.statusText.textContent = "Error";
    }

    // Show results bar
    els.resultsBar.style.display = "flex";
    els.resultExitCode.textContent = code;
    els.resultExitCode.style.color = code === 0 ? "var(--success)" : "var(--error)";
    els.resultDuration.textContent = formatTime(elapsed);
    els.resultStatus.textContent = code === 0 ? "Success" : "Failed";
    els.resultStatus.style.color = code === 0 ? "var(--success)" : "var(--error)";

    // Mark all stages done if success
    // Auto-advance raw capture folder for the next scan (e.g. 05 -> 06 -> 07)
    if (data.next_raw_capture) {
      const el = $("#cfgSaveRaw");
      if (el) {
        el.value = data.next_raw_capture;
        buildCommandPreview();
      }
    } else {
      const el = $("#cfgSaveRaw");
      if (el) {
        const m = el.value.match(/^(raw_capture_)(\d+)$/);
        if (m) {
          const nextNum = parseInt(m[2], 10) + 1;
          el.value = `${m[1]}${String(nextNum).padStart(2, "0")}`;
          buildCommandPreview();
        }
      }
    }

    // Reload scan history and raw captures dropdown
    loadScanHistory();
    loadRawCaptures();
  });

  // ================================================================
  // Button handlers
  // ================================================================

  els.btnStart.addEventListener("click", () => {
    if (isRunning) return;
    currentStage = 0;
    updateStageIndicators(0);
    socket.emit("start_scan", getConfig());
  });

  els.btnStop.addEventListener("click", () => {
    if (!isRunning) return;
    socket.emit("stop_scan");
  });

  els.btnClear.addEventListener("click", () => {
    els.console.innerHTML = "";
    welcomeCleared = false;
    els.resultsBar.style.display = "none";
    currentStage = 0;
    updateStageIndicators(0);
    els.elapsedTimer.textContent = "00:00";
    // Re-add welcome
    els.console.innerHTML = `
      <div class="console-welcome">
        <div class="welcome-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
          </svg>
        </div>
        <h3>Ready to Scan</h3>
        <p>Configure your scan parameters on the left, then press <strong>Start Scan</strong> to begin.</p>
        <p class="welcome-hint">The scanner uses Intel RealSense D455 to capture multi-view depth data and reconstruct 3D geometry.</p>
      </div>
    `;
  });

  els.btnScrollBottom.addEventListener("click", () => {
    els.console.scrollTop = els.console.scrollHeight;
  });

  // ================================================================
  // Preset buttons & Target select sync
  // ================================================================

  $$(".preset-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      applyPreset(btn.dataset.preset);
    });
  });

  const targetSelect = $("#cfgTarget");
  if (targetSelect) {
    targetSelect.addEventListener("change", () => {
      syncPresetButtons(targetSelect.value);
      buildCommandPreview();
    });
  }

  const saveRawInput = $("#cfgSaveRaw");
  if (saveRawInput) {
    saveRawInput.addEventListener("input", buildCommandPreview);
    saveRawInput.addEventListener("change", buildCommandPreview);
  }

  // ================================================================
  // Lightbox
  // ================================================================

  els.lightboxClose.addEventListener("click", () => {
    els.lightbox.classList.remove("open");
  });

  els.lightbox.addEventListener("click", (e) => {
    if (e.target === els.lightbox) els.lightbox.classList.remove("open");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && els.lightbox.classList.contains("open")) {
      els.lightbox.classList.remove("open");
    }
  });

  function openLightbox(src) {
    els.lightboxImg.src = src;
    els.lightbox.classList.add("open");
  }

  // ================================================================
  // Raw capture preview
  // ================================================================

  function loadRawCaptures() {
    fetch("/api/raw-captures")
      .then((r) => r.json())
      .then((captures) => {
        const sel = els.previewCapture;
        sel.innerHTML = '<option value="">Select...</option>';
        captures.forEach((name) => {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          sel.appendChild(opt);
        });
      })
      .catch(() => {});
  }

  els.previewCapture.addEventListener("change", () => {
    const name = els.previewCapture.value;
    if (!name) {
      els.thumbnailGrid.innerHTML = '<p class="thumbnail-placeholder">Select a capture directory to preview frames.</p>';
      return;
    }

    fetch(`/api/capture-thumbs/${name}`)
      .then((r) => r.json())
      .then((files) => {
        if (!files.length) {
          els.thumbnailGrid.innerHTML = '<p class="thumbnail-placeholder">No color frames found.</p>';
          return;
        }

        els.thumbnailGrid.innerHTML = "";
        files.forEach((fname) => {
          const card = document.createElement("div");
          card.className = "thumb-card";
          const imgSrc = `/raw-frame/${name}/${fname}`;
          card.innerHTML = `
            <img src="${imgSrc}" alt="${fname}" loading="lazy" />
            <span class="thumb-label">${fname.replace("_color.png", "")}</span>
          `;
          card.addEventListener("click", () => openLightbox(imgSrc));
          els.thumbnailGrid.appendChild(card);
        });
      })
      .catch(() => {
        els.thumbnailGrid.innerHTML = '<p class="thumbnail-placeholder">Failed to load frames.</p>';
      });
  });

  // ================================================================
  // Scan history
  // ================================================================

  function loadScanHistory() {
    fetch("/api/scan-history")
      .then((r) => r.json())
      .then((scans) => {
        if (!scans.length) {
          els.scanHistoryList.innerHTML = '<p class="thumbnail-placeholder">No scan history found.</p>';
          return;
        }

        els.scanHistoryList.innerHTML = "";
        scans.forEach((scan) => {
          const item = document.createElement("div");
          item.className = "history-item";

          let metaText = "";
          if (scan.measurement) {
            const m = scan.measurement;
            if (m.dimensions_mm) {
              const d = m.dimensions_mm;
              metaText = `${d.length?.toFixed(0) || "?"}×${d.width?.toFixed(0) || "?"}×${d.height?.toFixed(0) || "?"} mm`;
            }
          }

          item.innerHTML = `
            <div class="history-icon">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
              </svg>
            </div>
            <div class="history-info">
              <div class="history-name">${scan.name}</div>
              <div class="history-meta">${metaText || "Scan completed"}</div>
            </div>
          `;

          item.addEventListener("click", () => {
            // Set replay to this scan's raw capture dir if available
            appendLog(`[GUI] Selected scan: ${scan.name}`, "info");
          });

          els.scanHistoryList.appendChild(item);
        });
      })
      .catch(() => {
        els.scanHistoryList.innerHTML = '<p class="thumbnail-placeholder">Failed to load history.</p>';
      });
  }

  // ================================================================
  // Toggle config panel
  // ================================================================

  const toggleBtn = $("#toggleConfigBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const panel = $("#configPanel");
      panel.classList.toggle("collapsed");
      const svg = toggleBtn.querySelector("svg");
      if (panel.classList.contains("collapsed")) {
        svg.style.transform = "rotate(180deg)";
      } else {
        svg.style.transform = "";
      }
    });
  }

  // ================================================================
  // Init
  // ================================================================

  buildCommandPreview();
  fetchNextRawCapture();
  loadRawCaptures();
  loadScanHistory();

  // Keyboard shortcut: Ctrl+Enter to start
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter" && !isRunning) {
      els.btnStart.click();
    }
  });

})();
