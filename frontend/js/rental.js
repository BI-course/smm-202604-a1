/* rental.js -- page logic for the rental price model */

const panel = document.getElementById("result-panel");

connectForm({
  form: document.getElementById("model-form"),
  panel: panel,
  path: "/api/v1/models/rental-price-regressor/predictions",
  render: function (data) {
    showInPanel(
      panel,
      el("p", "result-label", "Estimated rent per month"),
      el("p", "result-figure", formatKes(data.predicted_rental_price_kes)),
      el("p", "result-meta", "Model: " + data.model)
    );
  },
});
