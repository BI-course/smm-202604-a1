/* attrition.js -- page logic for the employee attrition model */

const panel = document.getElementById("result-panel");

connectForm({
  form: document.getElementById("model-form"),
  panel: panel,
  path: "/api/v1/models/employee-attrition-classifier/predictions",
  render: function (data) {
    const chance = data.probability_of_leaving;
    const percent = (chance * 100).toFixed(1) + "%";

    // A bar that fills up to the chance of leaving
    const bar = el("div", "bar");
    bar.setAttribute("role", "img");
    bar.setAttribute("aria-label", "Chance of leaving: " + percent);
    const fill = el("div", "bar-fill");
    fill.style.width = percent;
    bar.appendChild(fill);

    showInPanel(
      panel,
      el("p", "result-label", "Likely to leave the company?"),
      el("p", "result-figure", data.predicted_attrition),
      bar,
      el("p", "bar-caption", "Chance of leaving: " + percent),
      el("p", "result-meta", "Model: " + data.model)
    );
  },
});
