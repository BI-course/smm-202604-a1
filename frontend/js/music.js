/* music.js -- page logic for the music recommender (association rules) */

const panel = document.getElementById("result-panel");
const form = document.getElementById("model-form");
const list = document.getElementById("track-list");

// The 16 tracks, grouped by genre. Each is [track id, name].
const GENRES = [
  ["Fitness / EDM",     [["M01", "Workout Anthem"], ["M02", "Pump It Up"], ["M03", "Cardio Beats"]]],
  ["Instrumental",      [["M04", "Study Focus 1"], ["M05", "Study Focus 2"]]],
  ["Ambient / Sleep",   [["M06", "Rain Sounds"], ["M07", "Deep Sleep"]]],
  ["Gospel",            [["M08", "Sunday Worship 1"], ["M09", "Sunday Worship 2"], ["M10", "Praise Medley"]]],
  ["R&B",               [["M11", "Heartbreak Ballad"], ["M12", "Love Story"]]],
  ["Hip-Hop",           [["M13", "Hip Hop Anthem"], ["M14", "Street Beats"]]],
  ["Amapiano / Afrobeat", [["M15", "Amapiano Vibes"], ["M16", "Gengetone Hit"]]],
];

// Quick examples that are known to match a saved rule.
const EXAMPLES = [
  ["Gospel pair", ["M09", "M10"]],
  ["Workout pair", ["M01", "M03"]],
  ["Hip-hop + Amapiano", ["M13", "M15"]],
];

// Draw the tick-boxes
for (const [genre, tracks] of GENRES) {
  const group = el("div", "track-group");
  group.appendChild(el("h3", "", genre));
  for (const [id, name] of tracks) {
    const row = el("label", "track");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = id;
    row.append(box, document.createTextNode(name + " (" + id + ")"));
    group.appendChild(row);
  }
  list.appendChild(group);
}

function setTicks(ids) {
  for (const box of list.querySelectorAll("input")) box.checked = ids.includes(box.value);
  document.getElementById("track-error").replaceChildren();
}

// Draw the example buttons
for (const [label, ids] of EXAMPLES) {
  const chip = el("button", "chip", label);
  chip.type = "button";
  chip.addEventListener("click", function () { setTicks(ids); });
  document.getElementById("examples").appendChild(chip);
}

function ticked() {
  return Array.from(list.querySelectorAll("input:checked")).map(function (b) { return b.value; });
}

function showRecommendations(data) {
  if (data.recommendations.length === 0) {
    const box = el("div", "notice notice-wait");
    box.appendChild(el("p", "", "No saved rule matches these tracks, so there is nothing to recommend. " +
      "Try a pair that is often played together, such as Sunday Worship 2 + Praise Medley."));
    showInPanel(panel, box);
    return;
  }
  const nodes = [el("p", "result-label", "Play next (after " + data.session_tracks.join(", ") + ")")];
  data.recommendations.forEach(function (rec, index) {
    const item = el("div", "rec");
    item.appendChild(el("p", "rec-name", (index + 1) + ". " + rec.item_name + " (" + rec.track_id + ")"));
    item.appendChild(el("p", "rec-genre", rec.category));
    item.appendChild(el("p", "bar-caption",
      "Confidence " + (rec.confidence * 100).toFixed(1) + "%, lift " + rec.lift));
    nodes.push(item);
  });
  nodes.push(el("p", "result-meta", "Model: " + data.model));
  showInPanel(panel, ...nodes);
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  // Check 1: at least one track picked
  const picked = ticked();
  const errorSlot = document.getElementById("track-error");
  errorSlot.replaceChildren();
  if (picked.length === 0) {
    const note = el("p", "field-error", "Tick at least one track.");
    errorSlot.appendChild(note);
    showError(panel, "Some values need fixing. They are marked in the form.");
    return;
  }
  // Check 2: the number of suggestions
  if (!checkNumbers(form)) {
    showError(panel, "Some values need fixing. They are marked in the form.");
    return;
  }

  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  showWaiting(panel);
  const result = await callApi("/api/v1/models/music-session-recommender/predictions", {
    session_tracks: picked,
    top_n: Number(document.getElementById("top_n").value),
  });
  button.disabled = false;

  if (result.ok) showRecommendations(result.data);
  else showError(panel, result.message);
});

form.addEventListener("reset", function () {
  document.getElementById("track-error").replaceChildren();
  clearFieldErrors(form);
  showInPanel(panel, el("p", "result-empty", "Your answer will appear here."));
});
