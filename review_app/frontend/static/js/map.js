/*
map.js
======
Leaflet map: Esri World Imagery basemap, reported-location marker + parcel
polygon, candidate markers + polygons (color-coded by rank), and click-to-
capture for the truth_outside_candidates verdict path.
*/

const ReviewMap = (() => {
    let map = null;
    let reportedLayer = null;
    let candidateLayers = [];       // [{layerGroup, ll_uuid, rank}]
    let truthMarker = null;
    let captureMode = false;
    let onTruthCaptured = null;     // callback(lat, lng)

    const CANDIDATE_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"];
    // rank 1 = red (most confident guess -- draw attention first), fading
    // through the palette to rank 5 = blue. Arbitrary but consistent.

    function init() {
        map = L.map("map").setView([39.5, -82.0], 7); // rough Ohio-ish default

        L.tileLayer(
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            {
                attribution: "Tiles &copy; Esri",
                maxZoom: 20,
            }
        ).addTo(map);

        map.on("click", (e) => {
            if (!captureMode) return;
            setTruthMarker(e.latlng.lat, e.latlng.lng);
            if (onTruthCaptured) onTruthCaptured(e.latlng.lat, e.latlng.lng);
        });

        return map;
    }

    function clearAll() {
        if (reportedLayer) { map.removeLayer(reportedLayer); reportedLayer = null; }
        candidateLayers.forEach(c => map.removeLayer(c.layerGroup));
        candidateLayers = [];
        clearTruthMarker();
    }

    function geojsonToLatLngBounds(geometry) {
        try {
            const layer = L.geoJSON(geometry);
            return layer.getBounds();
        } catch (e) {
            return null;
        }
    }

    function addReportedPoint(lat, lng, geometry, contextHtml) {
        const group = L.layerGroup();
        const marker = L.marker([lat, lng], {
            icon: L.divIcon({ className: "reported-marker", html: "&#9733;", iconSize: [24, 24] })
        }).bindPopup(contextHtml || "Reported location");
        group.addLayer(marker);

        if (geometry) {
            const poly = L.geoJSON(geometry, {
                style: { color: "#ffffff", weight: 2, fillOpacity: 0.1, dashArray: "4 4" }
            }).bindPopup(contextHtml || "Reported parcel");
            group.addLayer(poly);
        }

        group.addTo(map);
        reportedLayer = group;
        return group;
    }

    function addCandidate(rank, ll_uuid, geometry, popupHtml) {
        const color = CANDIDATE_COLORS[Math.min(rank - 1, CANDIDATE_COLORS.length - 1)];
        const group = L.layerGroup();

        if (geometry) {
            const poly = L.geoJSON(geometry, {
                style: { color, weight: 3, fillOpacity: 0.25 }
            }).bindPopup(popupHtml);
            group.addLayer(poly);

            const bounds = geojsonToLatLngBounds(geometry);
            if (bounds && bounds.isValid()) {
                const center = bounds.getCenter();
                const label = L.marker(center, {
                    icon: L.divIcon({
                        className: "candidate-rank-label",
                        html: `<div style="background:${color}">${rank}</div>`,
                        iconSize: [22, 22],
                    })
                }).bindPopup(popupHtml);
                group.addLayer(label);
            }
        }

        group.addTo(map);
        candidateLayers.push({ layerGroup: group, ll_uuid, rank });
        return group;
    }

    function fitToAllLayers() {
        const allBounds = [];
        if (reportedLayer) {
            reportedLayer.eachLayer(l => { if (l.getBounds) allBounds.push(l.getBounds()); });
        }
        candidateLayers.forEach(c => {
            c.layerGroup.eachLayer(l => { if (l.getBounds) allBounds.push(l.getBounds()); });
        });
        if (allBounds.length === 0) return;
        let combined = allBounds[0];
        allBounds.slice(1).forEach(b => { combined = combined.extend(b); });
        map.fitBounds(combined, { padding: [40, 40], maxZoom: 18 });
    }

    function setCaptureMode(enabled, callback) {
        captureMode = enabled;
        onTruthCaptured = callback || null;
        document.getElementById("map").style.cursor = enabled ? "crosshair" : "";
    }

    function setTruthMarker(lat, lng) {
        clearTruthMarker();
        truthMarker = L.marker([lat, lng], {
            icon: L.divIcon({ className: "truth-marker", html: "&#10060;", iconSize: [24, 24] })
        }).addTo(map).bindPopup("Marked true location").openPopup();
    }

    function clearTruthMarker() {
        if (truthMarker) { map.removeLayer(truthMarker); truthMarker = null; }
    }

    return {
        init, clearAll, addReportedPoint, addCandidate, fitToAllLayers,
        setCaptureMode, clearTruthMarker,
    };
})();
