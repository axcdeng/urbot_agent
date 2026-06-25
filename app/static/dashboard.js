async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

async function refreshAll() {
  const [robotState, locations, tasks, missions] = await Promise.all([
    fetchJson("/robot/status"),
    fetchJson("/locations"),
    fetchJson("/tasks"),
    fetchJson("/missions"),
  ]);
  document.getElementById("robot-state").textContent = JSON.stringify(robotState, null, 2);
  document.getElementById("locations").textContent = JSON.stringify(locations, null, 2);
  document.getElementById("tasks").textContent = JSON.stringify(tasks, null, 2);
  document.getElementById("missions").textContent = JSON.stringify(missions, null, 2);
}

async function sendCommand() {
  const input = document.getElementById("chat-input");
  const output = document.getElementById("chat-output");
  output.textContent = "Sending...";
  try {
    const payload = await fetchJson("/agent/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input.value }),
    });
    output.textContent = JSON.stringify(payload, null, 2);
    input.value = "";
    await refreshAll();
  } catch (error) {
    output.textContent = String(error);
  }
}

document.getElementById("refresh-button").addEventListener("click", refreshAll);
document.getElementById("chat-button").addEventListener("click", sendCommand);
document.getElementById("cancel-button").addEventListener("click", async () => {
  await fetchJson("/tasks/cancel", { method: "POST" });
  await refreshAll();
});
document.getElementById("estop-button").addEventListener("click", async () => {
  await fetchJson("/tasks/emergency-stop", { method: "POST" });
  await refreshAll();
});
refreshAll().catch((error) => {
  document.getElementById("robot-state").textContent = String(error);
});
setInterval(() => refreshAll().catch(() => {}), 5000);
