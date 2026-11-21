/* store.js -- page logic for the store segment (clustering) model */

const panel = document.getElementById("result-panel");

// What a typical store in each cluster looks like. These numbers come from
// the saved K-Means cluster centres (undoing the log and scaling steps), so
// they describe each segment in general, not one particular store.
// The names are our own short labels for what the numbers show.
const SEGMENTS = [
  { id: 0, name: "Small, quiet stores",             stores: 89,  revenue: 435615,  traffic: 172, basket: 86 },
  { id: 1, name: "Few shoppers, big baskets",       stores: 71,  revenue: 1044302, traffic: 142, basket: 246 },
  { id: 2, name: "Mid-sized, steady stores",        stores: 124, revenue: 1388848, traffic: 387, basket: 121 },
  { id: 3, name: "Top-revenue, high-traffic stores", stores: 40,  revenue: 4627202, traffic: 884, basket: 170 },
  { id: 4, name: "Busy stores, small baskets",      stores: 76,  revenue: 640411,  traffic: 659, basket: 53 },
];

function cell(text, className) {
  return el("td", className || "", text);
}

function profileTable(assignedId) {
  const table = el("table");
  const head = el("tr");
  for (const [text, cls] of [["Segment", ""], ["Revenue (KES)", "num"], ["Traffic", "num"], ["Basket (KES)", "num"]]) {
    head.appendChild(el("th", cls, text));
  }
  table.appendChild(head);

  for (const seg of SEGMENTS) {
    const row = el("tr", seg.id === assignedId ? "is-assigned" : "");
    row.appendChild(cell(seg.id + ". " + seg.name));
    row.appendChild(cell(seg.revenue.toLocaleString("en-US"), "num"));
    row.appendChild(cell(seg.traffic.toLocaleString("en-US"), "num"));
    row.appendChild(cell(seg.basket.toLocaleString("en-US"), "num"));
    table.appendChild(row);
  }
  return table;
}

connectForm({
  form: document.getElementById("model-form"),
  panel: panel,
  path: "/api/v1/models/store-performance-segmenter/predictions",
  render: function (data) {
    const id = data.assigned_cluster;
    const known = SEGMENTS.find(function (s) { return s.id === id; });
    showInPanel(
      panel,
      el("p", "result-label", "Assigned segment"),
      el("p", "result-figure", "Cluster " + id),
      el("p", "result-note", known ? known.name : "Unnamed segment"),
      el("p", "result-label", "Typical store in each segment (highlighted row is yours):"),
      profileTable(id),
      el("p", "result-meta", "Model: " + data.model)
    );
  },
});
