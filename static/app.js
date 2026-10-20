const controls = document.getElementById("controls");
const fileInput = document.getElementById("file");
const editor = document.getElementById("scenario");
const method = document.getElementById("method");
const message = document.getElementById("message");
const metrics = document.getElementById("metrics");
const tables = document.getElementById("tables");
const responseBlock = document.getElementById("response");
const jsonOutput = document.getElementById("json");
const costsTable = document.getElementById("costs-table");

const numbers = new Intl.NumberFormat("ru-RU", {
  maximumFractionDigits: 3,
});


function resetOutput(text = "") {
  message.textContent = text;
  message.className = "";
  metrics.textContent = "";
  tables.replaceChildren();
  jsonOutput.textContent = "";
  responseBlock.hidden = true;
}


function showError(text) {
  message.textContent = text;
  message.className = "error";
}


function addTable(title, headers, rows, target = tables) {
  const heading = document.createElement("h3");
  heading.textContent = title;

  const wrapper = document.createElement("div");
  wrapper.className = "table-wrap";

  const table = document.createElement("table");
  const head = table.createTHead().insertRow();

  for (const label of headers) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = label;
    head.append(cell);
  }

  const body = table.createTBody();

  for (const values of rows) {
    const row = body.insertRow();

    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");

      if (index === 0) {
        cell.scope = "row";
      }

      cell.textContent = typeof value === "number"
        ? numbers.format(value)
        : value;

      row.append(cell);
    });
  }

  wrapper.append(table);
  target.append(heading, wrapper);
}


function renderCostMatrix() {
  costsTable.replaceChildren();

  if (!editor.value.trim()) {
    const hint = document.createElement("p");
    hint.textContent = "Загрузи сценарий, чтобы увидеть матрицу тарифов.";
    costsTable.append(hint);
    return;
  }

  try {
    const scenario = JSON.parse(editor.value);

    if (!scenario || typeof scenario !== "object") {
      throw new Error("Ожидается объект сценария.");
    }

    const {depots, stations, costs} = scenario;

    if (!Array.isArray(depots) || depots.length === 0
        || !Array.isArray(stations) || stations.length === 0) {
      throw new Error("Нужны непустые массивы depots и stations.");
    }

    const validIds = [...depots, ...stations].every(
      item => item && typeof item.id === "string" && item.id.trim() !== "",
    );

    if (!validIds) {
      throw new Error("У каждой нефтебазы и АЗС должен быть id.");
    }

    const validCosts = Array.isArray(costs)
      && costs.length === depots.length
      && costs.every(row =>
        Array.isArray(row)
        && row.length === stations.length
        && row.every(value =>
          typeof value === "number"
          && Number.isFinite(value)
          && value >= 0,
        ),
      );

    if (!validCosts) {
      throw new Error("Проверь размеры costs и неотрицательность тарифов.");
    }

    addTable(
      "Матрица тарифов доставки, руб./м³",
      ["Нефтебаза / АЗС", ...stations.map(station => station.id)],
      depots.map((depot, i) => [depot.id, ...costs[i]]),
      costsTable,
    );
  } catch (error) {
    const hint = document.createElement("p");
    hint.textContent = "Таблица не показана: " + error.message;
    costsTable.append(hint);
  }
}


function showPlan(data) {
  message.textContent = data.message
    + "\nОбщая стоимость: "
    + numbers.format(data.total_cost)
    + " руб.";

  if (data.metrics) {
    const info = data.metrics;

    metrics.textContent = "Режим: " + info.method
      + "; время решателя: "
      + numbers.format(info.solver_seconds * 1000)
      + " мс; итерации: " + info.iterations
      + "; crossover: " + info.crossover_iterations + ".";
  }

  addTable(
    "План поставок, м³",
    ["Нефтебаза / АЗС", ...data.station_ids],
    data.depot_ids.map((id, i) => [id, ...data.deliveries[i]]),
  );

  addTable(
    "Нефтебазы, м³",
    ["Нефтебаза", "Отгружено", "Осталось"],
    data.depot_ids.map((id, i) => [
      id,
      data.summary.depot_shipped[i],
      data.summary.depot_remaining[i],
    ]),
  );

  addTable(
    "АЗС, м³",
    ["АЗС", "Поступило", "После поставки", "После потребления"],
    data.station_ids.map((id, j) => [
      id,
      data.summary.station_received[j],
      data.summary.station_after_delivery[j],
      data.summary.station_remaining[j],
    ]),
  );
}


async function sendRequest(action) {
  resetOutput("Обработка запроса…");
  controls.disabled = true;

  try {
    if (!editor.value.trim()) {
      throw new Error("Сначала загрузи или вставь сценарий JSON.");
    }

    const scenario = JSON.parse(editor.value);

    const url = action === "validate"
      ? "/api/validate"
      : "/api/optimize?method=" + encodeURIComponent(method.value);

    const response = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(scenario),
    });

    const data = await response.json();

    jsonOutput.textContent = JSON.stringify(data, null, 2);
    responseBlock.hidden = false;

    if (!response.ok) {
      throw new Error(data.message || "Ошибка HTTP " + response.status);
    }

    if (action === "validate") {
      if (data.feasible) {
        message.textContent = "Задача выполнима."
          + "\nЗапас нефтебаз: "
          + numbers.format(data.total_stock)
          + " м³."
          + "\nТребуется поставить: "
          + numbers.format(data.total_required)
          + " м³.";
      } else {
        showError(
          "Задача невыполнима:\n"
          + data.violations.map(item => item.message).join("\n"),
        );
      }
    } else if (data.status === "optimal") {
      showPlan(data);
    } else {
      showError("Статус: " + data.status + "\n" + data.message);
    }
  } catch (error) {
    showError("Ошибка: " + error.message);
  } finally {
    controls.disabled = false;
  }
}


fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];

  if (!file) {
    return;
  }

  resetOutput("Загрузка сценария…");
  controls.disabled = true;

  try {
    const scenario = JSON.parse(await file.text());

    editor.value = JSON.stringify(scenario, null, 2);
    message.textContent =
      "Сценарий загружен. Можно проверить данные или рассчитать план.";
  } catch (error) {
    editor.value = "";
    showError("Не удалось загрузить JSON: " + error.message);
  } finally {
    controls.disabled = false;
    fileInput.value = "";
    renderCostMatrix();
  }
});


editor.addEventListener("input", () => {
  resetOutput("Данные изменены. Выполни проверку или расчёт заново.");
});


method.addEventListener("change", () => {
  resetOutput("Режим изменён. Выполни расчёт заново.");
});


document.getElementById("validate").addEventListener("click", () => {
  sendRequest("validate");
});


document.getElementById("optimize").addEventListener("click", () => {
  sendRequest("optimize");
});

editor.addEventListener("input", renderCostMatrix);

renderCostMatrix();