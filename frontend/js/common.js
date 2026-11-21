/* =====================================================================
   common.js -- helpers shared by every page

   What is in here:
     1. Where the API lives (same server, or a different address)
     2. A small health check that fills in the "API status" label
     3. callApi()      -- sends JSON to the API and turns every failure
                          into a plain-English message
     4. readForm() / checkNumbers() -- read and check form fields
     5. small helpers for building the result panel safely
   ===================================================================== */

/* ---------------------------------------------------------------------
   1. Where is the API?
   - If this page is served by the same server as the API (Docker + Nginx,
     Render, or Flask itself), an empty string works: the browser calls
     the same address the page came from.
   - If the page is opened as a file, or from a simple static server
     (VS Code Live Server, PyCharm), the API is a different server, so we
     point at the local Flask address.
   - To point the pages at the Render API, put its address in
     API_ADDRESS_OVERRIDE, for example "https://your-app.onrender.com".
   --------------------------------------------------------------------- */
const API_ADDRESS_OVERRIDE = "";

function findApiAddress() {
  if (API_ADDRESS_OVERRIDE) return API_ADDRESS_OVERRIDE.replace(/\/+$/, "");
  const onlyAFile = window.location.protocol === "file:";
  const staticServerPorts = ["5500", "5501", "63342", "8080"];
  if (onlyAFile || staticServerPorts.includes(window.location.port)) {
    return "http://127.0.0.1:5000";
  }
  return "";
}
const API_ADDRESS = findApiAddress();

/* ---------------------------------------------------------------------
   2. API status label in the page header
   --------------------------------------------------------------------- */
async function showApiStatus() {
  const label = document.getElementById("api-status");
  if (!label) return;
  label.textContent = "Checking the API...";
  try {
    const reply = await fetch(API_ADDRESS + "/api/health");
    if (!reply.ok) throw new Error("bad status");
    label.dataset.state = "ok";
    label.textContent = "API is running";
  } catch (problem) {
    label.dataset.state = "down";
    label.textContent = "API is not reachable";
  }
}

/* ---------------------------------------------------------------------
   3. Talking to the API
   Always resolves to { ok: true, data } or { ok: false, message },
   so the pages never have to deal with exceptions.
   --------------------------------------------------------------------- */
async function callApi(path, payload) {
  let reply;
  try {
    reply = await fetch(API_ADDRESS + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (problem) {
    // The request never got an answer: server off, wrong address, no internet
    return {
      ok: false,
      message: "Could not reach the API. Check that the server is running, then try again.",
    };
  }

  // Try to read the body as JSON. A crashed server may send something else.
  let body = null;
  try {
    body = await reply.json();
  } catch (problem) {
    body = null;
  }

  if (!reply.ok) {
    const fromServer = body && body.error ? body.error : null;
    if (reply.status >= 500) {
      return { ok: false, message: fromServer || "The server had a problem. Please try again." };
    }
    return { ok: false, message: fromServer || "The API rejected this request (status " + reply.status + ")." };
  }
  if (body === null) {
    return { ok: false, message: "The API sent an answer we could not read." };
  }
  return { ok: true, data: body };
}

/* ---------------------------------------------------------------------
   4. Reading and checking form fields
   --------------------------------------------------------------------- */

// Collect the values of every named field in the form.
//  - number boxes give a number, or null when empty
//  - drop-downs give the selected text
function readForm(form) {
  const values = {};
  for (const field of form.querySelectorAll("input[name], select[name]")) {
    if (field.type === "number") {
      values[field.name] = field.value.trim() === "" ? null : Number(field.value);
    } else {
      values[field.name] = field.value;
    }
  }
  return values;
}

function clearFieldErrors(form) {
  for (const note of form.querySelectorAll(".field-error")) note.remove();
  for (const field of form.querySelectorAll(".invalid")) {
    field.classList.remove("invalid");
    field.removeAttribute("aria-invalid");
  }
}

function markFieldBad(field, message) {
  field.classList.add("invalid");
  field.setAttribute("aria-invalid", "true");
  const note = document.createElement("p");
  note.className = "field-error";
  note.textContent = message;
  field.closest(".field").appendChild(note);
}

// Check the form against the rules written on the inputs themselves:
//   required / min / max, and data-optional="true" for boxes that may be empty.
// Returns true when everything is fine. Otherwise marks the bad fields,
// moves the cursor to the first one, and returns false.
function checkNumbers(form) {
  clearFieldErrors(form);
  let firstBad = null;

  for (const field of form.querySelectorAll("input[type=number]")) {
    const label = form.querySelector('label[for="' + field.id + '"]').textContent.trim();
    const text = field.value.trim();
    const optional = field.dataset.optional === "true";
    let problem = null;

    if (text === "") {
      if (!optional) problem = "Please enter a value.";
    } else if (!Number.isFinite(Number(text))) {
      problem = "Please enter a number.";
    } else {
      const number = Number(text);
      if (field.min !== "" && number < Number(field.min)) {
        problem = label + " cannot be less than " + field.min + ".";
      } else if (field.max !== "" && number > Number(field.max)) {
        problem = label + " cannot be more than " + field.max + ".";
      }
    }

    if (problem) {
      markFieldBad(field, problem);
      if (!firstBad) firstBad = field;
    }
  }

  if (firstBad) {
    firstBad.focus();
    return false;
  }
  return true;
}

/* ---------------------------------------------------------------------
   5. Building the result panel
   We never put text from the server into innerHTML. We build elements and
   use textContent, so odd characters can never be run as code.
   --------------------------------------------------------------------- */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function showInPanel(panel, ...nodes) {
  panel.replaceChildren(...nodes);
}

function showWaiting(panel) {
  showInPanel(panel, el("p", "notice notice-wait", "Asking the model..."));
}

function showError(panel, message) {
  const box = el("div", "notice notice-error");
  box.setAttribute("role", "alert");
  box.appendChild(el("p", "", message));
  showInPanel(panel, box);
}

function formatKes(number) {
  return "KES " + Math.round(number).toLocaleString("en-US");
}

// Connects a form to the API. Each page calls this once.
//   form      the <form> element
//   panel     the element where the answer appears
//   path      the API address after the host, e.g. /api/v1/...
//   prepare   (optional) turns the form values into the JSON to send
//   render    receives the API answer and draws it in the panel
function connectForm({ form, panel, path, prepare, render }) {
  const button = form.querySelector('button[type="submit"]');

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!checkNumbers(form)) {
      showError(panel, "Some values need fixing. They are marked in the form.");
      return;
    }
    const payload = prepare ? prepare(readForm(form)) : readForm(form);

    button.disabled = true;
    showWaiting(panel);
    const result = await callApi(path, payload);
    button.disabled = false;

    if (result.ok) {
      render(result.data);
    } else {
      showError(panel, result.message);
    }
  });

  form.addEventListener("reset", () => {
    clearFieldErrors(form);
    showInPanel(panel, el("p", "result-empty", "Your answer will appear here."));
  });
}

document.addEventListener("DOMContentLoaded", showApiStatus);
