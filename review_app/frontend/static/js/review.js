/*
review.js
=========
Drives the review flow: load next plant -> render candidates/task -> collect
verdict -> submit -> load next. Two review_task branches, matching
10_build_review_queue.py's design (see that script's docstring):
  - candidate_pick   : reviewer picks from up to 5 ranked candidates, or says
                        the reported point was right, or says truth is outside
                        the whole list.
  - confirm_reported : nothing to rank -- reviewer confirms/rejects the
                        reported point directly. Still has the
                        truth_outside_candidates path (Phase 4 #1) since
                        Stage 1 could be wrong in a way the reviewer can spot
                        even with no candidates generated.
*/

let map;
let currentPlant = null;
let currentCandidates = [];
let selectedVerdict = null;
let selectedCandidate = null;   // {ll_uuid, candidate_rank}
let capturedTruth = null;       // {lat, lng}
let confirmationType = null;    // auto-determined, see setVerdict()

document.addEventListener("DOMContentLoaded", async () => {
    map = ReviewMap.init();
    loadReviewerName();
    await loadRounds();       // sets currentRound
    refreshStatus();
    refreshNavList();
    loadNextPlant();          // now safe, currentRound is set

    document.getElementById("submit-verdict-btn").addEventListener("click", submitVerdict);
    document.getElementById("reviewer-name").addEventListener("change", saveReviewerName);
});

function loadReviewerName() {
    const saved = localStorage.getItem("reviewer_name");
    if (saved) document.getElementById("reviewer-name").value = saved;
}
function saveReviewerName() {
    localStorage.setItem("reviewer_name", document.getElementById("reviewer-name").value);
}
function getReviewerName() {
    return document.getElementById("reviewer-name").value.trim();
}

async function refreshStatus() {
    const res = await fetch("/api/plants/status");
    const status = await res.json();
    const pct = status.total ? Math.round((status.reviewed / status.total) * 100) : 0;
    document.getElementById("progress-text").textContent =
        `${status.reviewed} / ${status.total} reviewed (${status.remaining} remaining)`;
    document.getElementById("progress-bar-fill").style.width = `${pct}%`;
}

async function loadNextPlant() {
    resetVerdictState();
    const res = await fetch(`/api/plants/next?review_round=${currentRound}`);
    const data = await res.json();

    if (data.done) {
        document.getElementById("plant-cwns-id").textContent = "Queue complete";
        document.getElementById("task-panel").innerHTML =
            "<p>No unreviewed plants remain in this round's queue.</p>";
        document.getElementById("verdict-panel").classList.add("hidden");
        ReviewMap.clearAll();
        resetDock();
        return;
    }

    currentPlant = data.plant;
    currentCandidates = data.candidates;
    renderPlant(data.plant, data.candidates, data.reported_geometry, data.reported_context);
    highlightCurrentInNav(data.plant.cwns_id);
}

async function loadPlantById(cwnsId) {
    resetVerdictState();
    const res = await fetch(`/api/plants/${cwnsId}`);
    if (!res.ok) return;
    const data = await res.json();
    currentPlant = data.plant;
    currentCandidates = data.candidates;
    renderPlant(data.plant, data.candidates, data.reported_geometry, data.reported_context);
    highlightCurrentInNav(data.plant.cwns_id);
}

let currentRound = null;

async function loadRounds() {
    const res = await fetch("/api/plants/rounds");
    const data = await res.json();
    const select = document.getElementById("round-select");
    select.innerHTML = "";
    data.rounds.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r;
        opt.textContent = `Round ${r}`;
        select.appendChild(opt);
    });
    currentRound = data.latest;
    select.value = currentRound;
    select.addEventListener("change", () => {
        currentRound = parseInt(select.value, 10);
        refreshNavList();
        refreshStatus();
        loadNextPlant();   // jump to next unreviewed plant in the newly selected round
    });
}

async function refreshNavList() {
    const res = await fetch(`/api/plants/list?review_round=${currentRound}`);
    const plants = await res.json();

    document.getElementById("nav-count").textContent =
        `(${plants.filter(p => p.reviewed).length} / ${plants.length})`;

    const groups = { holdout: [], uncertain: [], random: [] };
    plants.forEach(p => { if (groups[p.queue_slice]) groups[p.queue_slice].push(p); });

    const container = document.getElementById("plant-nav-list");
    container.innerHTML = "";
    for (const [slice, items] of Object.entries(groups)) {
        if (items.length === 0) continue;
        const label = document.createElement("div");
        label.className = "nav-group-label";
        label.textContent = `${slice} (${items.length})`;
        container.appendChild(label);

        items.forEach(p => {
            const row = document.createElement("div");
            row.className = "nav-item";
            row.dataset.cwnsId = p.cwns_id;
            const label = p.facility_name ? `${p.facility_name}` : p.cwns_id;
            row.title = p.facility_name ? `${p.facility_name} (${p.cwns_id})` : p.cwns_id;
            row.innerHTML = `
                <span class="nav-slice-dot ${p.queue_slice}"></span>
                <span class="nav-item-label">${label}</span>
                ${p.reviewed ? '<span class="nav-reviewed-badge">&#10003;</span>' : ''}
            `;
            row.addEventListener("click", () => loadPlantById(p.cwns_id));
            container.appendChild(row);
        });
    }
    if (currentPlant) highlightCurrentInNav(currentPlant.cwns_id);
}

function highlightCurrentInNav(cwnsId) {
    document.querySelectorAll(".nav-item").forEach(el => {
        el.classList.toggle("current", el.dataset.cwnsId === cwnsId);
    });
}

function renderPlant(plant, candidates, reportedGeometry, reportedContext) {
    document.getElementById("plant-cwns-id").textContent =
        plant.facility_name ? `${plant.facility_name} (CWNS ${plant.cwns_id})` : `CWNS ${plant.cwns_id}`;
    document.getElementById("plant-meta").innerHTML = `
        <div>State: ${plant.state_code}</div>
        <div>Stage 1 confidence (reported correct): ${
            plant.stage1_prob_correct !== null ? plant.stage1_prob_correct.toFixed(3) : "n/a (no parcel)"}</div>
        <div>Trigger: ${plant.trigger_reason}</div>
        <div>Queue slice: ${plant.queue_slice}</div>
        ${plant.owner_type ? `<div>Owner type: ${plant.owner_type}</div>` : ""}
    `;
    if (plant.reviewed) {
        document.getElementById("plant-meta").innerHTML +=
            `<div class="already-reviewed-note">Already reviewed by ${plant.reviewer} -- verdict: ${plant.plant_verdict}</div>`;
    }

    document.getElementById("holdout-badge").classList.toggle("hidden", !plant.is_holdout);
    renderDock(plant);

    ReviewMap.clearAll();
    if (plant.latitude && plant.longitude) {
        ReviewMap.addReportedPoint(plant.latitude, plant.longitude, reportedGeometry,
            buildReportedPopup(reportedContext));
    }

    const taskPanel = document.getElementById("task-panel");
    if (plant.review_task === "candidate_pick") {
        renderCandidatePickTask(candidates, taskPanel);
    } else {
        renderConfirmReportedTask(taskPanel);
    }

    ReviewMap.fitToAllLayers();
    renderVerdictButtons(plant.review_task, candidates.length > 0);
}

function renderDock(plant) {
    const identity = [plant.facility_name, plant.address, plant.city, plant.county_name, plant.zip_code]
        .filter(Boolean).join(", ");
    document.getElementById("dock-identity").textContent = identity || "Not available";

    document.getElementById("dock-population").textContent =
        plant.pop_served != null ? `${Math.round(plant.pop_served).toLocaleString()} residents served` : "Not available";

    const discharge = [];
    if (plant.surface_water_discharge) discharge.push("surface water discharge");
    if (plant.requires_npdes) discharge.push("requires NPDES");
    if (plant.any_reuse) discharge.push("reuse");
    document.getElementById("dock-discharge").textContent = discharge.length ? discharge.join(", ") : "None flagged";

    const geo = [plant.subdivision, plant.place, plant.county].filter(Boolean).join(" / ");
    document.getElementById("dock-geography").textContent =
        (geo || "Not available") + (plant.is_rural ? " (rural)" : "");
}

function resetDock() {
    ["dock-identity", "dock-population", "dock-discharge", "dock-geography"].forEach(id => {
        document.getElementById(id).textContent = "--";
    });
}

function renderCandidatePickTask(candidates, panel) {
    panel.innerHTML = "<h3>Candidates</h3><div id='candidate-list'></div>";
    const list = document.getElementById("candidate-list");

    candidates.forEach(c => {
        const popup = buildCandidatePopup(c);
        ReviewMap.addCandidate(c.candidate_rank, c.ll_uuid, c.geometry, popup);

        const row = document.createElement("div");
        row.className = "candidate-row";
        row.innerHTML = `
            <span class="rank-swatch rank-${c.candidate_rank}">${c.candidate_rank}</span>
            <span>score ${c.stage2a_score !== null ? c.stage2a_score.toFixed(3) : "n/a"}</span>
            <span>${c.distance_m !== null ? Math.round(c.distance_m) + "m" : ""}</span>
            <button data-uuid="${c.ll_uuid}" data-rank="${c.candidate_rank}" class="pick-candidate-btn">
                This is correct
            </button>
        `;
        list.appendChild(row);

        // owner/lbcs detail line under each row -- same info as the map
        // popup, visible without hovering, since the popup requires a click.
        const detail = document.createElement("div");
        detail.className = "candidate-detail";
        detail.innerHTML = buildCandidateDetailLine(c);
        list.appendChild(detail);
    });

    list.querySelectorAll(".pick-candidate-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            selectedCandidate = { ll_uuid: btn.dataset.uuid, candidate_rank: parseInt(btn.dataset.rank) };
            setVerdict("candidate_correct");
        });
    });
}

function buildCandidatePopup(c) {
    const lines = [`<b>Rank ${c.candidate_rank}</b>`, `Score: ${c.stage2a_score !== null ? c.stage2a_score.toFixed(3) : "n/a"}`];
    if (c.distance_m != null) lines.push(`Distance: ${Math.round(c.distance_m)}m`);
    if (c.owner) lines.push(`Owner: ${c.owner}`);
    if (c.lbcs_activity_desc) lines.push(`Activity: ${c.lbcs_activity_desc}`);
    if (c.lbcs_ownership_desc) lines.push(`Ownership: ${c.lbcs_ownership_desc}`);
    if (c.ll_gisacre != null) lines.push(`Size: ${c.ll_gisacre.toFixed(2)} acres`);
    if (c.zoning_type) lines.push(`Zoning: ${c.zoning_type}`);
    return lines.join("<br>");
}

function buildCandidateDetailLine(c) {
    const bits = [];
    if (c.owner) bits.push(c.owner);
    if (c.lbcs_activity_desc) bits.push(c.lbcs_activity_desc);
    if (c.ll_gisacre != null) bits.push(`${c.ll_gisacre.toFixed(1)} ac`);
    return bits.length ? bits.join(" &middot; ") : "<i>No parcel attribute data</i>";
}

function buildReportedPopup(ctx) {
    if (!ctx) return "Reported location";
    const lines = ["<b>Reported location</b>"];
    if (ctx.owner) lines.push(`Owner: ${ctx.owner}`);
    if (ctx.lbcs_activity_desc) lines.push(`Activity: ${ctx.lbcs_activity_desc}`);
    if (ctx.lbcs_ownership_desc) lines.push(`Ownership: ${ctx.lbcs_ownership_desc}`);
    if (ctx.ll_gisacre != null) lines.push(`Size: ${ctx.ll_gisacre.toFixed(2)} acres`);
    if (ctx.zoning_type) lines.push(`Zoning: ${ctx.zoning_type}`);
    if (lines.length === 1) lines.push("<i>No parcel attribute data (no parcel found here)</i>");
    return lines.join("<br>");
}

function renderConfirmReportedTask(panel) {
    panel.innerHTML = `
        <h3>No candidates generated</h3>
        <p>Stage 1 ${currentPlant.trigger_reason === "none" ? "passed this plant as correct" : "flagged this plant, but no candidate parcel survived filtering"}.
        Confirm whether the reported location (white marker/outline) is actually correct.</p>
    `;
}

function renderVerdictButtons(reviewTask, hasCandidates) {
    const container = document.getElementById("verdict-buttons");
    container.innerHTML = "";

    const btn = (label, verdict) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.className = "verdict-btn";
        b.dataset.verdict = verdict;
        b.addEventListener("click", () => setVerdict(verdict));
        return b;
    };

    container.appendChild(btn("Reported location is correct", "reported_correct"));
    container.appendChild(btn("Truth is not shown here (click map)", "truth_outside_candidates"));
    container.appendChild(btn("Needs more info / skip", "needs_info"));

    document.getElementById("verdict-panel").classList.remove("hidden");
    updateVerdictStatus();
}

function updateVerdictStatus() {
    const statusEl = document.getElementById("verdict-status");
    if (!statusEl) return;

    document.querySelectorAll(".verdict-btn").forEach(b => b.classList.remove("selected"));
    document.querySelectorAll(".pick-candidate-btn").forEach(b => b.classList.remove("selected"));

    if (!selectedVerdict) {
        statusEl.textContent = "No verdict selected yet.";
        statusEl.className = "";
        return;
    }

    if (selectedVerdict === "candidate_correct" && selectedCandidate) {
        statusEl.textContent = `Selected: Candidate rank ${selectedCandidate.candidate_rank} is correct. Choose a confirmation type below, then Submit.`;
        const pickBtn = document.querySelector(`.pick-candidate-btn[data-rank="${selectedCandidate.candidate_rank}"]`);
        if (pickBtn) pickBtn.classList.add("selected");
    } else {
        const labels = {
            reported_correct: "Selected: Reported location is correct.",
            truth_outside_candidates: "Selected: Truth is not shown here -- click the map to mark it.",
            needs_info: "Selected: Needs more info / skip.",
        };
        statusEl.textContent = labels[selectedVerdict] || "";
        const activeBtn = document.querySelector(`.verdict-btn[data-verdict="${selectedVerdict}"]`);
        if (activeBtn) activeBtn.classList.add("selected");
    }
    statusEl.className = "verdict-status-active";
}

function setVerdict(verdict) {
    selectedVerdict = verdict;
    capturedTruth = null;
    confirmationType = null;
    ReviewMap.setCaptureMode(false);
    document.getElementById("confirmation-type-display").classList.add("hidden");
    document.getElementById("truth-capture-panel").classList.add("hidden");

    // confirmation_type is DETERMINED, not chosen -- Phase 4's definition is
    // specifically about the model's TOP-RANKED pick, not "any shown
    // candidate". Asking reviewers to self-classify this proved confusing
    // in practice (2026-08-26): picking a non-top candidate left neither
    // radio option feeling correct, since "confirmed_proposal" only applies
    // to rank 1. This removes that judgment call entirely.
    if (verdict === "candidate_correct" && selectedCandidate) {
        confirmationType = selectedCandidate.candidate_rank === 1
            ? "confirmed_proposal" : "independent";
        const display = document.getElementById("confirmation-type-display");
        display.textContent = confirmationType === "confirmed_proposal"
            ? "Confirmation type: confirmed the model's top pick (rank 1)."
            : `Confirmation type: independent -- rank ${selectedCandidate.candidate_rank} was not the model's top pick.`;
        display.classList.remove("hidden");
    } else if (verdict === "reported_correct") {
        // The reported point isn't a Stage 2 candidate at all -- there is no
        // model "proposal" being confirmed here, only Stage 1 flagging
        // uncertainty. Always independent verification, not agreement with
        // a specific proposed answer.
        confirmationType = "independent";
        const display = document.getElementById("confirmation-type-display");
        display.textContent = "Confirmation type: independent (direct verification, not a candidate proposal).";
        display.classList.remove("hidden");
    } else if (verdict === "truth_outside_candidates") {
        document.getElementById("truth-capture-panel").classList.remove("hidden");
        document.getElementById("truth-coords").textContent = "No point selected yet.";
        ReviewMap.setCaptureMode(true, (lat, lng) => {
            capturedTruth = { lat, lng };
            document.getElementById("truth-coords").textContent =
                `Selected: ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
            updateSubmitEnabled();
        });
    }
    updateVerdictStatus();
    updateSubmitEnabled();
}

function updateSubmitEnabled() {
    const btn = document.getElementById("submit-verdict-btn");
    if (!selectedVerdict) { btn.disabled = true; return; }
    if (!getReviewerName()) { btn.disabled = true; return; }

    if (selectedVerdict === "candidate_correct" && !selectedCandidate) { btn.disabled = true; return; }
    if (selectedVerdict === "truth_outside_candidates" && !capturedTruth) { btn.disabled = true; return; }

    const needsConfirmation = selectedVerdict === "reported_correct" || selectedVerdict === "candidate_correct";
    if (needsConfirmation && !confirmationType) { btn.disabled = true; return; }

    btn.disabled = false;
}

document.addEventListener("input", (e) => {
    if (e.target.id === "reviewer-name") updateSubmitEnabled();
});

async function submitVerdict() {
    if (currentPlant && currentPlant.reviewed) {
        document.getElementById("submit-status").textContent =
            "This plant was already reviewed -- use the nav list to pick an unreviewed plant.";
        return;
    }

    const payload = {
        cwns_id: currentPlant.cwns_id,
        plant_verdict: selectedVerdict,
        reviewer: getReviewerName(),
        selected_ll_uuid: selectedCandidate ? selectedCandidate.ll_uuid : null,
        candidate_rank: selectedCandidate ? selectedCandidate.candidate_rank : null,
        truth_latitude: capturedTruth ? capturedTruth.lat : null,
        truth_longitude: capturedTruth ? capturedTruth.lng : null,
        confirmation_type: confirmationType,
        reviewer_notes: document.getElementById("reviewer-notes").value || null,
    };

    const statusEl = document.getElementById("submit-status");
    statusEl.textContent = "Submitting...";

    const res = await fetch("/api/verdict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });

    if (!res.ok) {
        const err = await res.json();
        statusEl.textContent = `Error: ${err.detail || res.statusText}`;
        return;
    }

    statusEl.textContent = "";
    await refreshStatus();
    await refreshNavList();
    await loadNextPlant();
}

function resetVerdictState() {
    selectedVerdict = null;
    selectedCandidate = null;
    capturedTruth = null;
    confirmationType = null;
    ReviewMap.setCaptureMode(false);
    document.getElementById("reviewer-notes").value = "";
    document.getElementById("submit-status").textContent = "";
    document.getElementById("confirmation-type-display").classList.add("hidden");
    document.getElementById("truth-capture-panel").classList.add("hidden");
    document.getElementById("submit-verdict-btn").disabled = true;
    const statusEl = document.getElementById("verdict-status");
    if (statusEl) { statusEl.textContent = "No verdict selected yet."; statusEl.className = ""; }
}
