//configuration for dynamically processing index page
//loads either the basic or advance html

//used static files
//advanced.html : template html for displaying all the actions + runtime parameter input
//basic.html : template html for displaying a minimal view of actions

//on page reload get saved data
window.onload = async function () {
  initStaticMeta();
  try {
    await loadBasicOrAdvanced();
  } catch (error) {
    showViewError(error);
  }
  initStickyTables();
  initResultsTabs();
  refreshLastRunBanner();
  loadSystemOverviewBadges();

  //add listener for basic and advanced html switch
  const switchBtn = document.getElementById("basicOrAdvanced");
  if (switchBtn) {
    switchBtn.addEventListener("click", () => SwitchBasicOrAdvanced());
  }
};

//action buttons of the advanced view
const ADVANCED_ACTION_IDS = [
  "dayahead-optim",
  "forecast-model-fit",
  "forecast-model-predict",
  "forecast-model-tune",
  "forecast-calibration",
  "regressor-model-fit",
  "regressor-model-predict",
  "export-influxdb-to-csv",
  "perfect-optim",
  "publish-data",
  "naive-mpc-optim",
];

//elements the view needs once loaded (loadButtons, the loader status, the saved inputs), per view
const VIEW_CONTROL_IDS = {
  basic: ["dayahead-optim-basic", "loader"],
  advanced: [...ADVANCED_ACTION_IDS, "input-plus", "input-minus", "input-select", "input-clear", "loader", "input-container"],
};
//id of the first element of each template, which identifies the displayed view
const VIEW_ROOT_ID = { basic: "basic", advanced: "advance" };

//add listeners to buttons (based on page)
async function loadButtons(page) {
  switch (page) {
    case "advanced":
      ADVANCED_ACTION_IDS.forEach((id) =>
        document
          .getElementById(id)
          .addEventListener("click", () => formAction(id, "advanced"))
      );
      ["input-plus", "input-minus"].forEach((id) =>
        document
          .getElementById(id)
          .addEventListener("click", () => dictInputs(id))
      );
      document
        .getElementById("input-select")
        .addEventListener("change", () => getSavedData());
      document
        .getElementById("input-clear")
        .addEventListener("click", () => ClearInputData());
      
      // Check if InfluxDB is enabled and show/hide export section
      checkInfluxDBAndShowExport(); //not awaited: it handles its own errors and must not delay the view setup
      initAdvancedTabs();
      initRuntimePresets();
      const thermalCalibBtn = document.getElementById("thermal-model-calibration");
      if (thermalCalibBtn) {
        thermalCalibBtn.addEventListener("click", () => runThermalModelCalibration());
      }
      break;
    case "basic":
      document
        .getElementById("dayahead-optim-basic")
        .addEventListener("click", () => formAction("dayahead-optim", "basic"));
      break;
  }
  updateModeBadge(page);
}

const VIEW_ERROR_PREFIX = "Unable to load the page view: ";

//a template or export-check request still pending after this long fails instead of leaving the page waiting
//(browsers without AbortSignal.timeout keep waiting, as before). Written without `?.`: this page's script has no
//ES2020 syntax (optional chaining), and a parse error would break the whole page, not just the timeout.
const REQUEST_TIMEOUT_MS = 30000;

//fetch the basic or advanced html and replace the current view, only if it holds the controls it needs
async function loadView(page) {
  const file = page + ".html";
  const htmlData = await getHTMLData(file);
  const probe = document.createElement("template");
  probe.innerHTML = htmlData;
  const missing = VIEW_CONTROL_IDS[page].filter((id) => probe.content.querySelector(`#${id}`) === null);
  const root = probe.content.firstElementChild;
  if (root === null || root.id !== VIEW_ROOT_ID[page]) {
    missing.unshift(VIEW_ROOT_ID[page]);
  }
  if (missing.length) {
    throw new Error(`${file} is incomplete (missing ${missing.join(", ")})`);
  }
  document.getElementById("TabSelection").innerHTML = htmlData; //container element to house basic or advanced data
  await loadButtons(page); //load buttons based on basic or advanced
  if (document.getElementById("alert-text").textContent.includes(VIEW_ERROR_PREFIX)) {
    document.getElementById("alert").style.display = "none"; //only a failed earlier view load is recovered, not another action's error
  }
}

//on check present basic or advanced html inside form element
async function loadBasicOrAdvanced(RequestedPage) {
  //first check any function  arg
  if (arguments.length == 1) {
    switch (RequestedPage) {
      case "basic":
        await loadView("basic");
        if (testStorage()) {
          localStorage.setItem("TabSelection", "basic");
        } //remember mode (save to localStorage)
        return "basic"; //return basic to get saved data
      case "advanced":
        await loadView("advanced");
        if (testStorage()) {
          localStorage.setItem("TabSelection", "advanced");
        }
        getSavedData();
        return "advanced";
      default:
        await loadView("advanced");
        getSavedData();
        return "advanced";
    }
  }
  //then check localStorage
  if (testStorage()) {
    if (
      localStorage.getItem("TabSelection") !== null &&
      localStorage.getItem("TabSelection") === "advanced"
    ) {
      //if advance
      await loadView("advanced");
      getSavedData();
      return "advanced";
    } else {
      //else run basic (first time)
      await loadView("basic");
      return "basic";
    }
  } else {
    //if localStorage not supported, set to advanced page
    await loadView("advanced");
    return "advanced";
  }
}

//on button press, check current displayed page data and switch
let switching = false; //a click while a switch is still loading is ignored, so loads cannot overlap
async function SwitchBasicOrAdvanced() {
  if (switching) {
    return;
  }
  switching = true;
  try {
    const current = document.getElementById("TabSelection").firstElementChild;
    await loadBasicOrAdvanced(current && current.id === "basic" ? "advanced" : "basic");
  } catch (error) {
    showViewError(error); //the current view is left untouched
  } finally {
    switching = false;
  }
}

//report a failed view load in the alert box (the view's own status icons may not exist)
function showViewError(error) {
  document.getElementById("alert-text").textContent =
    "\r\n\u2022 " + VIEW_ERROR_PREFIX + error.message;
  document.getElementById("alert").style.display = "block";
  document.getElementById("alert").style.textAlign = "left";
}

//get html data from basic.html or advanced.html
async function getHTMLData(htmlFile) {
  //revalidate: the template changes with the release, and loadView checks it against this script
  const response = await fetch(`static/` + htmlFile, {
    cache: "no-cache",
    signal: AbortSignal.timeout && AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`${htmlFile} request failed with HTTP ${response.status}`);
  }
  let blob = await response.blob(); //get data blob
  let htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  return htmlTemplateData;
}

//function pushing data via post, triggered by button action
async function formAction(action, page) {
  let data = {}
  if (page !== "basic") {
    //dont try to get input data in basic mode
    data = inputToJson();
  } else {
    data = {};
  } //send no data

  if (data !== 0) {
    //don't run if there is an error in the input (box/list) Json data
    showChangeStatus("loading", {}); // show loading div for status
    let actionCompleted = false;
    try {
      const response = await fetch(`action/` + action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (response.ok) {
        actionCompleted = true;
        if (page !== "basic") {
          saveStorage();
        }
        await getTemplate();
        showChangeStatus(response.status, []);
        return true;
      }
      let errorPayload;
      try {
        errorPayload = await response.json();
      } catch (_) {
        errorPayload = [`Request failed with HTTP ${response.status}`];
      }
      if (!Array.isArray(errorPayload)) {
        errorPayload = [`Request failed with HTTP ${response.status}`];
      }
      showChangeStatus(response.status, errorPayload);
      return false;
    } catch (error) {
      const message = error.message || "Request failed";
      showChangeStatus("request-error", [actionCompleted
        ? `Action completed, but updating the results failed: ${message}`
        : message]);
      return false;
    } finally {
      document.getElementById("loader").classList.remove("loading");
    }
  } else {
    showChangeStatus("remove"); //replace loading, show tick or cross with none
    return false;
  }
}

//function in control of status icons of post above
function showChangeStatus(status, logJson) {
  let loading = document.getElementById("loader"); //element showing statuses
  const alert = document.getElementById("alert");
  if (alert && (status === "loading" || (typeof status === "number" && status >= 200 && status < 300))) {
    alert.style.display = "none";
  }
  if (status === "remove") {
    //remove all
    loading.innerHTML = "";
    loading.classList.remove("loading");
  } else if (status === "loading") {
    //show loading logo
    loading.innerHTML = "";
    loading.classList.add("loading"); //append class with loading animation styling
  } else if (typeof status === "number" && status >= 200 && status < 300) {
    // Match response.ok: any successful HTTP response shows a tick.
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=tick>&#x2713;</p>`;
  } else {
    //then show a cross
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=cross>&#x292C;</p>`; //show cross icon to indicate an error
    if (logJson.length != 0 && document.getElementById("alert-text") !== null) {
      document.getElementById("alert-text").textContent =
        "\r\n\u2022 " + logJson.join("\r\n\u2022 "); //show received log data in alert box
      document.getElementById("alert").style.display = "block";
      document.getElementById("alert").style.textAlign = "left";
    }
  }
}

//get rendered html template with containing new table data
async function getTemplate() {
  //fetch data from webserver.py
  let htmlTemplateData = "";
  let response = await fetch(`template`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(`Template refresh failed with HTTP ${response.status}`);
  }
  let blob = await response.blob(); //get data blob
  htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  let templateDiv = document.getElementById("template"); //get template container element to override
  clearStickyTables();
  templateDiv.innerHTML = htmlTemplateData; //override container inner html with new data
  let scripts = Array.from(templateDiv.getElementsByTagName("script")); //replace script tags manually
  for (const script of scripts) {
    let TempScript = document.createElement("script");
    TempScript.innerHTML = script.innerHTML;
    script.parentElement.appendChild(TempScript);
  }
  if (
    document.getElementById("results-tabs") &&
    !document.getElementById("stage-times-client-fallback") &&
    typeof document.createElement === "function" &&
    typeof templateDiv.appendChild === "function"
  ) {
    const fallbackDiv = document.createElement("div");
    fallbackDiv.id = "stage-times-client-fallback";
    fallbackDiv.className = "table_div";
    fallbackDiv.setAttribute("data-plot-key", "figure_stage_times_client");
    fallbackDiv.style.display = "none";
    templateDiv.appendChild(fallbackDiv);
  }
  if (typeof document.querySelectorAll === "function") {
    initStickyTables();
  }
  initResultsTabs();
  refreshLastRunBanner();
}

let stickyTableCleanups = [];

function clearStickyTables() {
  stickyTableCleanups.forEach((cleanup) => cleanup());
  stickyTableCleanups = [];
}

function initStickyTables() {
  clearStickyTables();

  document.querySelectorAll("table.mystyle").forEach((table) => {
    const thead = table.querySelector("thead");
    if (!thead) return;
    const container = table.closest(".table_div");
    if (!container) return;

    // Fixed wrapper clips the clone to the container's visible horizontal area
    const wrapper = document.createElement("div");
    wrapper.className = "sticky-header-wrapper";
    document.body.appendChild(wrapper);

    // Clone table containing only the header
    const cloneTable = document.createElement("table");
    cloneTable.className = table.className;
    cloneTable.style.tableLayout = "fixed";
    cloneTable.appendChild(thead.cloneNode(true));
    wrapper.appendChild(cloneTable);

    // Sync wrapper position and column widths with the live table
    const syncGeometry = () => {
      const r = container.getBoundingClientRect();
      const tableRect = table.getBoundingClientRect();
      const left = r.left + container.clientLeft;
      wrapper.style.left = left + "px";
      wrapper.style.width = container.clientWidth + "px";
      cloneTable.style.width = tableRect.width + "px";
      cloneTable.style.margin = `0 0 0 ${tableRect.left - left + container.scrollLeft}px`;
      thead.querySelectorAll("th").forEach((th, i) => {
        const cloneTh = cloneTable.querySelectorAll("th")[i];
        if (cloneTh) {
          cloneTh.style.width = th.getBoundingClientRect().width + "px";
        }
      });
      syncHScroll();
      updateVisibility();
    };
    window.addEventListener("resize", syncGeometry, { passive: true });

    // Horizontal scroll: sync wrapper.scrollLeft — avoids transform stacking context,
    // so overflow:hidden clips correctly on both sides
    const syncHScroll = () => {
      wrapper.scrollLeft = container.scrollLeft;
      const firstTh = cloneTable.querySelector("thead th:first-child");
      if (firstTh) {
        firstTh.style.transform = `translateX(${wrapper.scrollLeft}px)`;
      }
    };
    container.addEventListener("scroll", syncHScroll, { passive: true });

    // Vertical scroll: toggle visibility only — no per-frame transform update
    let shown = null;
    const updateVisibility = () => {
      const shouldShow =
        thead.getBoundingClientRect().top <= 0 &&
        table.getBoundingClientRect().bottom > 0;
      if (shouldShow !== shown) {
        shown = shouldShow;
        wrapper.style.visibility = shouldShow ? "visible" : "hidden";
      }
    };
    window.addEventListener("scroll", updateVisibility, { passive: true });
    const frame = requestAnimationFrame(syncGeometry);
    stickyTableCleanups.push(() => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", syncGeometry);
      window.removeEventListener("scroll", updateVisibility);
      container.removeEventListener("scroll", syncHScroll);
      wrapper.remove();
    });
  });
}

//test localStorage support
function testStorage() {
  try {
    localStorage.setItem("test", { test: "123" });
    localStorage.removeItem("test");
    return true;
  } catch (error) {
    return false;
  }
}

//function gets saved data (if any)
function getSavedData() {
  dictInputs(); //check selected current (List or Box) is correct
  if (testStorage()) {
    //if local storage exists and works
    let selectElement = document.getElementById("input-select"); // select button element
    let input_container = document.getElementById("input-container"); // container div containing all dynamic input elements (Box/List)
    if (
      localStorage.getItem("input_container_content") &&
      localStorage.getItem("input_container_content") !== "{}"
    ) {
      //If items already stored in local storage, then override default
      if (selectElement.value == "Box") {
        //if Box is selected, show saved json data into box
        document.getElementById("text-area").value = localStorage.getItem(
          "input_container_content"
        );
      }
      if (selectElement.value == "List") {
        //if List is selected, show saved json data into box
        let storedJson = {};
        try {
          storedJson = JSON.parse(localStorage.getItem("input_container_content")) || {};
        } catch (error) {
          console.error("Ignoring invalid saved runtime data:", error); //the view is still usable without it
        }
        if (Object.keys(storedJson).length > 0) {
          input_container.innerHTML = "";
          for (const ikey in storedJson) {
            input_container.appendChild(
              createInputListDiv(ikey, JSON.stringify(storedJson[ikey]))
            ); //call function to present each key as an list div element (with saved values)
          }
        }
      }
    }
  }
}

//using localStorage, store json data from input-list(List)/text-area(from input-box) elements for saved state save on page refresh (will save state on successful post)
function saveStorage() {
  let data = JSON.stringify(inputToJson());
  if (testStorage() && data != "{}") {
    //don't bother saving if empty and/or storage don't exist
    localStorage.setItem("input_container_content", data);
  }
}

//function gets values from input-list/text-area(from input-box) elements and return json dict object
function inputToJson() {
  let input_container = document.getElementById("input-container"); //container
  let inputListArr = document.getElementsByClassName("input-list"); //list
  let inputTextArea = document.getElementById("text-area"); //box
  let input_container_child = null;
  input_container_child = input_container.firstElementChild; //work out which element is first inside container div
  let jsonReturnData = {};

  if (input_container_child == null) {
    //if no elements in container then return empty
    return jsonReturnData;
  }
  //if List return box json
  if (
    input_container_child.className == "input-list" &&
    inputListArr.length > 0
  ) {
    //if list is first and if list is greater then 0, otherwise give empty dict

    let jsonTempData = "{";
    for (let i = 0; i < inputListArr.length; i++) {
      let key = inputListArr[i].getElementsByClassName("input-key")[0].value;
      let value =
        inputListArr[i].getElementsByClassName("input-value")[0].value;
      //curate a string with list elements to parse into json later
      if (key !== "") {
        //key must not be empty
        if (i !== 0) {
          jsonTempData = jsonTempData.concat(",");
        } //add comma before every parameter, exuding the first
        jsonTempData = jsonTempData.concat('"' + key + '":' + value);
      }
    }
    jsonTempData = jsonTempData.concat("}");
    try {
      jsonReturnData = JSON.parse(jsonTempData);
    } catch (error) {
      //if json error, show in alert box
      document.getElementById("alert-text").textContent =
        "\r\n" +
        error +
        "\r\n" +
        "JSON Error: String values may not be wrapped in quotes";
      document.getElementById("alert").style.display = "block";
      document.getElementById("alert").style.textAlign = "center";
      return 0;
    }
  }
  //if Box return box json
  if (
    input_container_child.className == "input-box" &&
    inputTextArea.value != ""
  ) {
    //if Box is first and text is not empty, otherwise give empty dict
    try {
      jsonReturnData = JSON.parse(inputTextArea.value);
    } catch (error) {
      //if json error, show in alert box
      document.getElementById("alert-text").textContent = "\r\n" + error;
      document.getElementById("alert").style.display = "block";
      return 0;
    }
  }
  return jsonReturnData;
}

//function creates input list div element (and pass it values if given)
function createInputListDiv(ikey, ivalue) {
  let div = document.createElement("div");
  div.className = "input-list";
  div.innerHTML = `
                <input class="input-key" type="text" placeholder="pv_power_forecast" >
                <p>:</p>
                <input class="input-value" type="text" placeholder="[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 70, 141.22, 246.18, 513.5, 753.27, 1049.89, 1797.93, 1697.3, 3078.93, 1164.33, 1046.68, 1559.1, 2091.26, 1556.76, 1166.73, 1516.63, 1391.13, 1720.13, 820.75, 804.41, 251.63, 79.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]" >
            `;

  if (ikey && ivalue) {
    //if value and key is provided (from local storage) then add as elements values
    div.getElementsByClassName("input-key")[0].value = String(ikey);
    div.getElementsByClassName("input-value")[0].value = String(ivalue);
  }

  return div;
}

//function assigned to control (add and remove) input (Box and List) elements
function dictInputs(action) {
  let input_container = document.getElementById("input-container"); // container div containing all dynamic input elements
  let selectElement = document.getElementById("input-select"); // select button
  let input_container_child = null;
  let input_container_child_name = null;
  if (input_container.children.length > 0) {
    input_container_child = input_container.firstElementChild; // figure out what is the first element inside of container (ie: "text-area" (input-box) or "input-list" (list))
    input_container_child_name = input_container.firstElementChild.className;
  }
  //if list is selected, remove text-area (from Box) element and replace (with input-list)
  if (selectElement.value == "List") {
    if (action == "input-plus" || input_container_child_name == "input-box") {
      //if plus button pressed, or Box element exists
      if (input_container_child_name == "input-box") {
        input_container_child.remove();
      }
      input_container.appendChild(createInputListDiv(false, false)); //call to createInputListDiv function to craft input-list element (with no values) and append inside container element
    }
    if (action == "input-minus") {
      //minus button pressed, remove input-list element
      if (input_container.children.length > 0) {
        let inputListArr = document.getElementsByClassName("input-list");
        let obj = inputListArr.item(inputListArr.length - 1);
        obj.innerHTML = "";
        obj.remove();
      }
    }
  }
  //if box is selected, remove input-list elements and replace (with text-area)
  if (selectElement.value == "Box") {
    if (
      input_container_child_name == "input-list" ||
      input_container_child === null
    ) {
      // if input list exists or no Box element
      input_container.innerHTML = ""; //remove input-list list elements via erasing container innerHTML
      let div = document.createElement("div"); //add input-box element
      div.className = "input-box";
      div.innerHTML = `
                <textarea id="text-area" rows="30" placeholder="{}"></textarea>
                `;
      input_container.appendChild(div); //append inside of container element
    }
  }
}

//clear stored input data from localStorage (if any), clear input elements
async function ClearInputData(id) {
  if (
    testStorage() &&
    localStorage.getItem("input_container_content") !== null
  ) {
    localStorage.setItem("input_container_content", "{}");
  }
  ClearInputElements();
}

//clear input elements
async function ClearInputElements() {
  let selectElement = document.getElementById("input-select");
  let input_container = document.getElementById("input-container");
  if (selectElement.value == "Box") {
    document.getElementById("text-area").value = "{}";
  }
  if (selectElement.value == "List") {
    input_container.innerHTML = "";
  }
}

// //Run day ahead, then publish actions
// async function DayheadOptimPublish() {
//     response = await formAction("dayahead-optim", "basic")
//     if (response) { //if successful publish data
//         formAction("publish-data", "basic")
//     }
//}

// Check if InfluxDB is configured and show/hide export section
let influxChecks = 0; //only the newest check may update the export section
async function checkInfluxDBAndShowExport() {
  const thisCheck = ++influxChecks;
  try {
    // relative, like every other URL here: behind Home Assistant ingress the app is served under a prefix
    const response = await fetch("get-config", {
      signal: AbortSignal.timeout && AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    if (response.ok) {
      const config = await response.json();
      if (thisCheck !== influxChecks) {
        return; //superseded by a newer check
      }
      renderSystemOverviewBadges(config);
      const exportSection = document.getElementById("export-influxdb-section");
      if (exportSection) {
        // Show export if InfluxDB is configured (has host and port)
        // Users can export from InfluxDB even if use_influxdb is false for optimization
        const isInfluxDBConfigured = 
          config.influxdb_host && 
          config.influxdb_host !== "" && 
          config.influxdb_port;
        
        if (isInfluxDBConfigured) {
          exportSection.style.display = "block";
        } else {
          exportSection.style.display = "none";
        }
      }
    }
  } catch (error) {
    console.error("Error checking InfluxDB configuration:", error);
  }
}

function initStaticMeta() {
  const yearEl = document.getElementById("current-year");
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }
}

function updateModeBadge(mode) {
  const labelEl = document.getElementById("mode-pill-label");
  if (labelEl) {
    labelEl.textContent = mode === "basic" ? "Basic" : "Advanced";
  }
}

function initAdvancedTabs() {
  const buttons = Array.prototype.slice.call(
    document.querySelectorAll(".action-tab-btn")
  );
  const panels = Array.prototype.slice.call(
    document.querySelectorAll(".action-tab-panel")
  );
  if (!buttons.length || !panels.length) {
    return;
  }
  buttons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const target = btn.getAttribute("data-action-tab");
      buttons.forEach(function (b) {
        b.classList.toggle("active", b === btn);
      });
      panels.forEach(function (panel) {
        const match = panel.getAttribute("data-action-panel") === target;
        panel.classList.toggle("active", match);
        panel.style.display = match ? "block" : "none";
      });
    });
  });
}

const RUNTIME_PRESETS = {
  mpc_horizon: {
    prediction_horizon: 10,
    soc_init: 0.45,
    soc_final: 0.5,
  },
  pv_quantiles: {
    pv_power_forecast: [0, 0, 120, 480, 1100, 1850, 2400, 2200, 1500, 700, 150, 0],
    pv_power_forecast_p10: [0, 0, 80, 320, 780, 1300, 1750, 1600, 1050, 460, 90, 0],
  },
  deferrable_window: {
    operating_hours_of_each_deferrable_load: [4, 2],
    start_timesteps_of_each_deferrable_load: [2, 0],
    end_timesteps_of_each_deferrable_load: [20, 16],
  },
  raw_json: {},
};

function initRuntimePresets() {
  const presetSelect = document.getElementById("runtime-preset-select");
  if (!presetSelect) {
    return;
  }
  presetSelect.addEventListener("change", function () {
    const key = presetSelect.value;
    if (!key || !Object.prototype.hasOwnProperty.call(RUNTIME_PRESETS, key)) {
      return;
    }
    const modeSelect = document.getElementById("input-select");
    if (modeSelect) {
      modeSelect.value = "Box";
      dictInputs();
      const box = document.getElementById("text-area");
      if (box) {
        box.value = JSON.stringify(RUNTIME_PRESETS[key], null, 2);
        saveStorage();
        if (key === "raw_json" && testStorage()) {
          localStorage.setItem("input_container_content", "{}");
        }
      }
    }
    presetSelect.value = "";
  });
}

const RESULTS_TAB_GROUPS = {
  schedule: [
    "title",
    "subsubtitle0",
    "figure_0",
    "figure_pv_forecast",
    "figure_thermal",
    "figure_1",
    "figure_battery_sources",
    "figure_2",
    "figure_calibration_metrics",
    "figure_thermal_lag_comparison",
    "figure_thermal_scatter",
  ],
  analytics: [
    "analytics_intro",
    "figure_energy_pie",
    "figure_histograms",
    "figure_heatmap",
    "figure_load_effectiveness",
  ],
  performance: ["figure_stage_times", "figure_stage_times_client"],
  tables: ["subsubtitle1", "table1", "subsubtitle2", "table2"],
};

function getResultsTabForKey(plotKey) {
  if (!plotKey) {
    return "schedule";
  }
  if (plotKey.indexOf("figure_fed_") === 0) {
    return "federated";
  }
  const categories = Object.keys(RESULTS_TAB_GROUPS);
  for (let i = 0; i < categories.length; i++) {
    const cat = categories[i];
    if (RESULTS_TAB_GROUPS[cat].indexOf(plotKey) !== -1) {
      return cat;
    }
  }
  return "schedule";
}

function resizeVisiblePlotlyPlots(container) {
  if (!container) {
    return;
  }
  const plotlyDivs = container.querySelectorAll(".js-plotly-plot");
  if (typeof window !== "undefined" && window.Plotly && window.Plotly.Plots && plotlyDivs.length) {
    for (let i = 0; i < plotlyDivs.length; i++) {
      try {
        window.Plotly.Plots.resize(plotlyDivs[i]);
      } catch (_err) {
        // ignore detached or hidden plots
      }
    }
  }
}

let activeResultsTab = "schedule";

function initResultsTabs() {
  const tabBar = document.getElementById("results-tabs");
  const templateDiv = document.getElementById("template");
  if (
    !tabBar ||
    !templateDiv ||
    typeof tabBar.querySelectorAll !== "function" ||
    typeof templateDiv.querySelectorAll !== "function"
  ) {
    return;
  }
  const items = Array.prototype.slice.call(
    templateDiv.querySelectorAll(".table_div[data-plot-key]")
  );
  const counts = {
    schedule: 0,
    analytics: 0,
    federated: 0,
    performance: 0,
    tables: 0,
    all: 0,
  };

  items.forEach(function (item) {
    const key = item.getAttribute("data-plot-key") || "";
    if (
      key === "figure_stage_times_client" &&
      (!item.innerHTML || !item.innerHTML.trim())
    ) {
      return;
    }
    const isCountableItem =
      key.indexOf("figure_") === 0 || key.indexOf("table") === 0;
    const tab = getResultsTabForKey(key);
    if (isCountableItem && Object.prototype.hasOwnProperty.call(counts, tab)) {
      counts[tab] += 1;
      counts.all += 1;
    }
  });

  const buttons = Array.prototype.slice.call(
    tabBar.querySelectorAll(".results-tab-btn")
  );
  buttons.forEach(function (btn) {
    const tab = btn.getAttribute("data-results-tab");
    const countSpan = btn.querySelector(".tab-count");
    const c = Object.prototype.hasOwnProperty.call(counts, tab) ? counts[tab] : 0;
    if (countSpan) {
      countSpan.textContent = c > 0 ? String(c) : "";
    }
    if (tab === "federated" || tab === "performance" || tab === "analytics") {
      btn.style.display = c > 0 ? "" : "none";
    }
  });

  const isSpecialReport =
    items.length > 0 &&
    !templateDiv.querySelector('[data-plot-key="figure_2"]') &&
    counts.tables > 0 &&
    counts.schedule > 0;
  if (isSpecialReport) {
    activeResultsTab = "all";
  } else if (counts[activeResultsTab] === 0 && activeResultsTab !== "all") {
    activeResultsTab = counts.schedule > 0 ? "schedule" : "all";
  }
  updateOptimStatusFromDom(templateDiv);

  function applyTabSelection(selectedTab) {
    activeResultsTab = selectedTab;
    const currentButtons = Array.prototype.slice.call(
      tabBar.querySelectorAll(".results-tab-btn")
    );
    const currentItems = Array.prototype.slice.call(
      templateDiv.querySelectorAll(".table_div[data-plot-key]")
    );
    currentButtons.forEach(function (btn) {
      btn.classList.toggle(
        "active",
        btn.getAttribute("data-results-tab") === selectedTab
      );
    });
    currentItems.forEach(function (item) {
      const key = item.getAttribute("data-plot-key") || "";
      if (
        key === "figure_stage_times_client" &&
        (!item.innerHTML || !item.innerHTML.trim())
      ) {
        item.style.display = "none";
        return;
      }
      const itemTab = getResultsTabForKey(key);
      const show = selectedTab === "all" || itemTab === selectedTab;
      item.style.display = show ? "" : "none";
    });
    resizeVisiblePlotlyPlots(templateDiv);
    if (typeof document.querySelectorAll === "function") {
      initStickyTables();
    }
  }

  tabBar._applyTabSelection = applyTabSelection;
  buttons.forEach(function (btn) {
    if (!btn.dataset.boundResultsTab) {
      btn.dataset.boundResultsTab = "1";
      btn.addEventListener("click", function () {
        if (typeof tabBar._applyTabSelection === "function") {
          tabBar._applyTabSelection(
            btn.getAttribute("data-results-tab") || "schedule"
          );
        }
      });
    }
  });

  applyTabSelection(activeResultsTab);
}

function renderStageTimesFallback(lastRun) {
  const fallbackEl = document.getElementById("stage-times-client-fallback");
  if (!fallbackEl) {
    return;
  }
  const hasServerPlot = Boolean(
    document.querySelector('#template [data-plot-key="figure_stage_times"]')
  );
  if (
    hasServerPlot ||
    !lastRun ||
    !lastRun.stage_times ||
    typeof lastRun.stage_times !== "object"
  ) {
    fallbackEl.innerHTML = "";
    fallbackEl.style.display = "none";
    return;
  }

  const entries = Object.keys(lastRun.stage_times)
    .map(function (k) {
      return { name: k, val: Number(lastRun.stage_times[k]) };
    })
    .filter(function (item) {
      return Number.isFinite(item.val) && item.val >= 0;
    });

  if (!entries.length) {
    fallbackEl.innerHTML = "";
    fallbackEl.style.display = "none";
    return;
  }

  const maxVal = entries.reduce(function (m, item) {
    return item.val > m ? item.val : m;
  }, 0.0001);
  const totalSec =
    typeof lastRun.duration_total_seconds === "number"
      ? lastRun.duration_total_seconds
      : entries.reduce(function (sum, item) {
          return item.name.indexOf(".") === -1 ? sum + item.val : sum;
        }, 0);

  let html = '<div class="stage-times-card">';
  html +=
    "<h3>Pipeline Stage Execution Times (total: " +
    totalSec.toFixed(3) +
    " s)</h3>";
  html += '<div class="stage-bars">';
  entries.forEach(function (entry) {
    const pct = Math.max(2, Math.min(100, (entry.val / maxVal) * 100));
    const ms = Math.round(entry.val * 1000);
    html += '<div class="stage-bar-row">';
    html += '<span class="stage-label">' + entry.name + "</span>";
    html +=
      '<div class="stage-track"><div class="stage-fill" style="width:' +
      pct.toFixed(1) +
      '%"></div></div>';
    html +=
      '<span class="stage-value">' +
      entry.val.toFixed(3) +
      " s (" +
      ms +
      " ms)</span>";
    html += "</div>";
  });
  html += "</div></div>";
  fallbackEl.innerHTML = html;
}

async function refreshLastRunBanner() {
  if (typeof fetch !== "function") {
    return;
  }
  if (
    !document.getElementById("last-run-pill") &&
    !document.getElementById("emhass-version-badge") &&
    !document.getElementById("stage-times-client-fallback")
  ) {
    return;
  }
  try {
    const opts =
      typeof AbortSignal !== "undefined" && AbortSignal.timeout
        ? { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) }
        : {};
    const response = await fetch("api/v1/last-run", opts);
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    const versionBadge = document.getElementById("emhass-version-badge");
    if (versionBadge && data && data.emhass_version) {
      versionBadge.textContent = "EMHASS v" + String(data.emhass_version);
    }
    const statusPill = document.getElementById("last-run-pill");
    if (statusPill && data) {
      statusPill.classList.remove("status-ok", "status-warn", "status-error");
      if (data.status === "ok" && !data.infeasible) {
        const dur =
          typeof data.duration_total_seconds === "number"
            ? " • " + data.duration_total_seconds.toFixed(2) + "s"
            : "";
        const act = data.action ? String(data.action) + dur : "OK" + dur;
        statusPill.textContent = act;
        statusPill.classList.add("status-ok");
      } else if (data.infeasible) {
        statusPill.textContent = "Infeasible";
        statusPill.classList.add("status-warn");
      } else if (data.status === "error") {
        statusPill.textContent = "Last run error";
        statusPill.classList.add("status-error");
      } else {
        statusPill.textContent = "No run recorded";
      }
    }
    const optimPill = document.getElementById("optim-status-pill");
    if (optimPill && data && (!optimPill.textContent || optimPill.style.display === "none")) {
      optimPill.classList.remove("status-ok", "status-warn", "status-error");
      if (data.status === "ok" && !data.infeasible) {
        optimPill.textContent = "Solver: Optimal";
        optimPill.classList.add("status-ok");
        optimPill.style.display = "";
      } else if (data.infeasible) {
        optimPill.textContent = "Solver: Infeasible";
        optimPill.classList.add("status-warn");
        optimPill.style.display = "";
      } else if (data.status === "error") {
        optimPill.textContent = "Solver: Error";
        optimPill.classList.add("status-error");
        optimPill.style.display = "";
      }
    }
    renderStageTimesFallback(data);
    initResultsTabs();
  } catch (_error) {
    // non-fatal status poll
  }
}

function updateOptimStatusFromDom(templateDiv) {
  const optimPill = document.getElementById("optim-status-pill");
  if (!optimPill || !templateDiv) {
    return;
  }
  const summaryDiv = templateDiv.querySelector('[data-plot-key="table2"]');
  if (!summaryDiv) {
    return;
  }
  const rows = summaryDiv.querySelectorAll("tr");
  for (let i = 0; i < rows.length; i++) {
    const cells = rows[i].querySelectorAll("td");
    if (cells.length >= 2 && cells[0].textContent.trim() === "optim_status") {
      const val = cells[1].textContent.trim();
      optimPill.classList.remove("status-ok", "status-warn", "status-error");
      optimPill.textContent = "Solver: " + val;
      if (val.indexOf("Optimal") === 0) {
        optimPill.classList.add("status-ok");
      } else if (val.indexOf("Infeasible") !== -1) {
        optimPill.classList.add("status-warn");
      } else {
        optimPill.classList.add("status-error");
      }
      optimPill.style.display = "";
      return;
    }
  }
}

function renderSystemOverviewBadges(config) {
  const strip = document.getElementById("system-overview-badges");
  if (!strip || !config || typeof config !== "object") {
    return;
  }
  const badges = [];
  if (config.set_use_pv) {
    const wf = config.weather_forecast_method ? " (" + String(config.weather_forecast_method) + ")" : "";
    badges.push({ cls: "sys-pv", text: "PV Plant" + wf });
  } else {
    badges.push({ cls: "sys-muted", text: "No PV" });
  }

  if (config.set_use_battery) {
    const nb = Number(config.number_of_batteries) || 1;
    badges.push({ cls: "sys-batt", text: nb > 1 ? nb + " Batteries" : "1 Battery" });
  } else {
    badges.push({ cls: "sys-muted", text: "No Battery" });
  }

  const ndef = Number(config.number_of_deferrable_loads);
  const defCount = Number.isFinite(ndef) ? ndef : 0;
  badges.push({
    cls: defCount > 0 ? "sys-loads" : "sys-muted",
    text: defCount + " Deferrable Load" + (defCount === 1 ? "" : "s"),
  });

  if (Array.isArray(config.def_load_config)) {
    let thermalCount = 0;
    for (let i = 0; i < config.def_load_config.length; i++) {
      const item = config.def_load_config[i];
      if (item && (item.thermal_config || item.thermal_battery)) {
        thermalCount += 1;
      }
    }
    if (thermalCount > 0) {
      badges.push({
        cls: "sys-thermal",
        text: thermalCount + " Thermal Load" + (thermalCount === 1 ? "" : "s"),
      });
    }
  }

  const nInv = Number(config.number_of_inverters) || 1;
  badges.push({
    cls: "sys-neutral",
    text: nInv > 1 ? nInv + " Inverters" : config.inverter_is_hybrid ? "Hybrid Inverter" : "1 Inverter",
  });

  const capCost = Array.isArray(config.capacity_cost_per_kw)
    ? config.capacity_cost_per_kw.some(function (v) { return Number(v) > 0; })
    : Number(config.capacity_cost_per_kw) > 0;
  if (capCost) {
    badges.push({ cls: "sys-tariff", text: "Capacity Tariff" });
  }
  if (config.costfun) {
    badges.push({ cls: "sys-neutral", text: "Objective: " + String(config.costfun) });
  }
  if (config.load_forecast_method) {
    badges.push({ cls: "sys-load", text: "Load: " + String(config.load_forecast_method) });
  }
  if (config.optimization_time_step) {
    badges.push({ cls: "sys-neutral", text: String(config.optimization_time_step) + " min step" });
  }
  if (config.optimization_backend === "dantzig_wolfe") {
    badges.push({ cls: "sys-fed", text: "Federated Coordinator" });
  }

  let html = '<span class="system-overview-title">Configured system:</span>';
  for (let j = 0; j < badges.length; j++) {
    html += '<span class="sys-badge ' + badges[j].cls + '">' + badges[j].text + "</span>";
  }
  strip.innerHTML = html;
}

async function loadSystemOverviewBadges() {
  if (typeof fetch !== "function" || !document.getElementById("system-overview-badges")) {
    return;
  }
  try {
    const opts =
      typeof AbortSignal !== "undefined" && AbortSignal.timeout
        ? { signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) }
        : {};
    const response = await fetch("get-config", opts);
    if (response.ok) {
      renderSystemOverviewBadges(await response.json());
    }
  } catch (_err) {
    // non-fatal badge fetch
  }
}

async function runThermalModelCalibration() {
  const payload = {};
  const extra = inputToJson();
  if (extra === 0) {
    return;
  }
  if (extra && typeof extra === "object") {
    Object.keys(extra).forEach(function (k) {
      payload[k] = extra[k];
    });
  }
  const numField = function (id, key, isInt) {
    const el = document.getElementById(id);
    if (el && el.value !== "") {
      const parsed = isInt ? parseInt(el.value, 10) : parseFloat(el.value);
      if (Number.isFinite(parsed)) {
        payload[key] = parsed;
      }
    }
  };
  const strField = function (id, key) {
    const el = document.getElementById(id);
    if (el && el.value && el.value.trim()) {
      payload[key] = el.value.trim();
    }
  };
  numField("thermal-tune-load-index", "load_index", true);
  numField("thermal-tune-p-nom", "nominal_power", false);
  numField("thermal-tune-lookback-hours", "lookback_hours", true);
  numField("thermal-tune-current-alpha", "current_heating_rate", false);
  numField("thermal-tune-current-gamma", "current_cooling_constant", false);
  numField("thermal-tune-current-inertia", "current_thermal_inertia", false);
  strField("thermal-tune-entity-tin", "entity_t_in");
  strField("thermal-tune-entity-tout", "entity_t_out");
  strField("thermal-tune-entity-power", "entity_power");
  strField("thermal-tune-entity-setpoint", "entity_t_pred");
  strField("thermal-tune-entity-switch", "entity_switch");

  const startEl = document.getElementById("thermal-tune-date-start");
  const endEl = document.getElementById("thermal-tune-date-end");
  if (startEl && endEl && startEl.value.trim() && endEl.value.trim()) {
    payload.date_ranges = [[startEl.value.trim(), endEl.value.trim()]];
  }

  showChangeStatus("loading");
  try {
    const response = await fetch("action/thermal-model-calibration", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.ok) {
      const loaderEl = document.getElementById("loader");
      if (loaderEl) {
        loaderEl.classList.remove("loading");
      }
      await getTemplate();
      showChangeStatus(response.status, {});
    } else {
      let msg = "";
      try {
        msg = await response.text();
      } catch (_e) {
        msg = "HTTP " + String(response.status);
      }
      showChangeStatus(response.status, [msg]);
    }
  } catch (error) {
    showChangeStatus(0, [String(error)]);
  }
}
