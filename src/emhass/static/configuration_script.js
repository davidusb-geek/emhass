//javascript file for dynamically processing configuration page

//used static files
//param_definitions.json : stores information about parameters (E.g. their defaults, their type, and what parameter section to be in)
//configuration_list.html : template html to act as a base for the list view. (Params get dynamically added after)

//Div layout
/* <div configuration-container>
  <div class="section-card">
    <div class="section-card-header"> POSSIBLE HEADER INPUT HERE WITH PARAMETER ID</div>
    <div class="section-body">
      <div id="PARAMETER-NAME" class="param">
        <div class="param-input">input/s here</div>
      </div>
    </div>
  </div>
</div>; */

//#610: the 15 per-battery array params in the "Battery" section (plant_conf +
//optim_conf, see utils.py BATT_ARRAY_PARAMS_PLANT_CONF/BATT_ARRAY_PARAMS_OPTIM_CONF/
//BATT_WEIGHT_PARAMS). Kept as an explicit list (not "every array.* param in the
//Battery section") because number_of_batteries itself is int (not array.*), so a
//generic filter would already exclude it - this list exists so the
//"number_of_batteries" header case below only grows/shrinks these, never the
//section's other array.* siblings if any get added later.
const BATTERY_ARRAY_PARAMS = [
  "weight_battery_discharge",
  "weight_battery_charge",
  "battery_discharge_power_max",
  "battery_charge_power_max",
  "battery_discharge_efficiency",
  "battery_charge_efficiency",
  "battery_nominal_energy_capacity",
  "battery_minimum_state_of_charge",
  "battery_maximum_state_of_charge",
  "battery_target_state_of_charge",
  "battery_stress_cost",
  "battery_soc_deficit_threshold",
  "battery_soc_deficit_cost",
  "battery_soc_surplus_threshold",
  "battery_soc_surplus_cost",
];

//#1116: the per-load array params in the "Deferrable Loads" section whose
//cardinality is driven by number_of_deferrable_loads (mirrors
//BATTERY_ARRAY_PARAMS/number_of_batteries above). heat_topology is
//deliberately excluded - it is a single object param (input: "object"), not
//a per-load array.* value, so it must stay exactly one input regardless of
//number_of_deferrable_loads.
const DEFERRABLE_ARRAY_PARAMS = [
  "nominal_power_of_deferrable_loads",
  "minimum_power_of_deferrable_loads",
  "cost_forecast_per_deferrable_load",
  "is_electric_load",
  "operating_hours_of_each_deferrable_load",
  "treat_deferrable_load_as_semi_cont",
  "set_deferrable_load_single_constant",
  "set_deferrable_startup_penalty",
  "deferrable_load_max_cost",
  "set_deferrable_max_startups",
  "def_minimum_on_time",
  "def_minimum_off_time",
  "start_timesteps_of_each_deferrable_load",
  "end_timesteps_of_each_deferrable_load",
];

//#1150: the per-inverter list params in the "System" section whose length is
//driven by number_of_inverters (mirrors BATTERY_ARRAY_PARAMS/number_of_batteries
//above). They are the same lists as INVERTER_CAP_PARAMS and
//INVERTER_BROADCAST_PARAMS in utils.py.
//The two power curves are left out because each is one JSON field holding the
//whole list of curves, and the two index lists because battery_inverter_index
//has one entry per battery and pv_plant_inverter_index one per PV plant.
const INVERTER_CAP_PARAMS = ["inverter_ac_output_max", "inverter_ac_input_max"];
const INVERTER_BROADCAST_PARAMS = [
  "inverter_efficiency_dc_ac",
  "inverter_efficiency_ac_dc",
  "inverter_stress_cost",
  "compute_curtailment",
];
const INVERTER_ARRAY_PARAMS = [...INVERTER_CAP_PARAMS, ...INVERTER_BROADCAST_PARAMS];
//the inverter lists follow number_of_inverters only up to this count, so a typo
//such as 1000000 cannot add millions of inputs and hang the page. The backend
//has no upper bound on number_of_inverters; 100 matches MAX_BATTERIES in utils.py
const INVERTER_SYNC_LIMIT = 100;
//the two power curves, editable above one inverter whatever inverter_is_hybrid
//says (see updateInverterCurveAccess)
const INVERTER_CURVE_PARAMS = ["inverter_power_curve_dc_ac", "inverter_power_curve_ac_dc"];

const INVERTER_CARD_PARAMS = new Set([
  "maximum_power_from_grid",
  "maximum_power_to_grid",
  "number_of_inverters",
  "inverter_is_hybrid",
  ...INVERTER_ARRAY_PARAMS,
  ...INVERTER_CURVE_PARAMS,
  "inverter_stress_segments",
  "battery_inverter_index",
  "pv_plant_inverter_index",
]);

let lastLoadedConfig = null;

//a request still pending after this long fails with an error instead of leaving the page waiting
//(browsers without AbortSignal.timeout keep waiting, as before)
const REQUEST_TIMEOUT_MS = 30000;

//on page reload
window.onload = async function () {
  ///fetch configuration parameters from definitions json file
  let param_definitions = await getParamDefinitions();
  //obtain configuration from emhass (pull)
  let config = param_definitions && (await obtainConfig());
  //obtain configuration_list.html html as a template to dynamically to render parameters in a list view (parameters as input items)
  let list_html = config && (await getListHTML());
  //fail closed: a failed load is not an empty configuration. Render no form, allow no save, say how to recover
  if (!list_html) {
    document.getElementById("save").disabled = true;
    document.getElementById("alert-text").textContent += "\r\n\u2022 Reload the page to try again.";
    return;
  }
  //load list parameter page (default)
  loadConfigurationListView(param_definitions, config, list_html);

  //add event listener to save button
  document
    .getElementById("save")
    .addEventListener("click", () => saveConfiguration(param_definitions));

  //add event listener to yaml button (convert yaml to json in box view)
  document.getElementById("yaml").addEventListener("click", () => yamlToJson());
  //hide yaml button by default (display in box view)
  document.getElementById("yaml").style.display = "none";

  //add event listener to defaults button
  document
    .getElementById("defaults")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, true)
    );

  //add event listener to json-toggle button (toggle between json box and list view)
  document
    .getElementById("json-toggle")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, false)
    );

  const yearEl = document.getElementById("current-year");
  if (yearEl) {
    yearEl.textContent = String(new Date().getFullYear());
  }
  const searchInput = document.getElementById("config-search-input");
  if (searchInput) {
    searchInput.addEventListener("input", () => applyConfigFilters());
  }
  const levelSelect = document.getElementById("config-level-filter");
  if (levelSelect) {
    levelSelect.addEventListener("change", () => applyConfigFilters());
  }
};

//obtain file containing information about parameters (definitions)
//returns null when the definitions could not be obtained
async function getParamDefinitions() {
  try {
    const response = await fetch(`static/data/param_definitions.json`, {
      cache: "no-cache", //revalidate: the definitions change with the release
      signal: AbortSignal.timeout?.(REQUEST_TIMEOUT_MS),
    });
    if (response.status !== 200 && response.status !== 201) {
      throw new Error(`HTTP ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    //alert error in alert box
    errorAlert(`Unable to obtain definitions file: ${error.message}`);
    return null;
  }
}

//request a configuration from emhass: the saved one or the defaults
//returns null when it could not be obtained (never an empty config)
async function fetchConfig(url, what) {
  config = {};
  try {
    const response = await fetch(url, {
      method: "GET",
      signal: AbortSignal.timeout?.(REQUEST_TIMEOUT_MS),
    });
    let response_status = response.status; //return status
    //if request failed
    if (response_status !== 200 && response_status !== 201) {
      showChangeStatus(response_status, await readErrorLog(response));
      return null;
    }
    //else extract json rom data
    let blob = await response.blob(); //get data blob
    config = await new Response(blob).json(); //obtain json from blob
    //an empty object would render the parameter defaults as if they were the saved settings
    if (config === null || typeof config !== "object" || Array.isArray(config) || Object.keys(config).length === 0) {
      throw new Error("the response is not a non-empty configuration object");
    }
    showChangeStatus(response_status, {});
    return config;
  } catch (error) {
    showChangeStatus(0, [`Unable to obtain ${what}: ${error.message}`]);
    return null;
  }
}

//obtain emhass config (from saved params extracted/simplified into the config format)
async function obtainConfig() {
  return fetchConfig(`get-config`, "the configuration");
}

//obtain emhass default config (to present the default parameters in view)
async function ObtainDefaultConfig() {
  return fetchConfig(`get-config/defaults`, "the default configuration");
}

//error messages of a failed response: the JSON list sent by emhass, else the HTTP status
async function readErrorLog(response) {
  try {
    const log = await response.json();
    if (Array.isArray(log) && log.length) {
      return log;
    }
  } catch (error) {
    //not JSON (e.g. a proxy error page)
  }
  return [`HTTP ${response.status}`];
}

//get html data from configuration_list.html (list template)
//returns null when the template could not be obtained
async function getListHTML() {
  try {
    const response = await fetch(`static/configuration_list.html`, {
      cache: "no-cache", //revalidate: the template changes with the release
      signal: AbortSignal.timeout?.(REQUEST_TIMEOUT_MS),
    });
    if (response.status !== 200 && response.status !== 201) {
      throw new Error(`HTTP ${response.status}`);
    }
    let blob = await response.blob(); //get data blob
    const html = await new Response(blob).text(); //obtain html from blob
    //an empty template cannot render a form: report it instead of failing closed without a message
    if (!html.trim()) {
      throw new Error("the file is empty");
    }
    return html;
  } catch (error) {
    errorAlert(`Unable to obtain configuration_list.html file: ${error.message}`);
    return null;
  }
}

//load list configuration view
function loadConfigurationListView(param_definitions, config, list_html) {
  if (list_html == null || config == null || param_definitions == null) {
    return 1;
  }
  lastLoadedConfig = config;

  //list parameters used in the section headers
  //#610: number_of_batteries added to auto-sync the 15 per-battery array params
  //(mirrors number_of_deferrable_loads), see the "number_of_batteries" case in headerElement
  //(number_of_inverters is not a header input: its lists follow it through setupInverterSync)
  let header_input_list = ["set_use_battery", "set_use_pv", "number_of_deferrable_loads", "number_of_batteries"];

  //get the main container and append list template html
  document.getElementById("configuration-container").innerHTML = list_html;

  //loop through configuration sections ('Local','System','Tariff','Solar System (PV)') in definitions file
  for (let section in param_definitions) {
    // build each section by adding parameters with their corresponding input elements
    buildParamContainers(
      section,
      param_definitions[section],
      config,
      header_input_list
    );

    //after sections have been built, add event listeners for section header inputs
    //loop though headers
    for (let header_input_param of header_input_list) {
      if (param_definitions[section].hasOwnProperty(header_input_param)) {
        //grab default from definitions file
        let value = param_definitions[section][header_input_param]["default_value"];
        //find input element (using the parameter name as the input element ID)
        let header_input_element = document.getElementById(header_input_param);
        if (header_input_element !== null) {
          //add event listener to element (trigger on input change)
          header_input_element.addEventListener("input", (e) =>
            headerElement(e.target, param_definitions, config)
          );
          //check the EMHASS config to see if it contains a stored param value
          //else keep default
          value = checkConfigParam(value, config, header_input_param);
          //set value of input
          header_input_element.value = value;
          //checkboxes (for Booleans) also set value to "checked"
          if (header_input_element.type == "checkbox") {
            header_input_element.checked = value;
          }
          //manually trigger the header parameter input event listener for setting up initial section state
          headerElement(header_input_element, param_definitions, config);
        }
      }
    }
  }

  //#1150: size the per-inverter lists to number_of_inverters
  setupInverterSync(param_definitions, config);

  setupDynamicVisibility();
  initConfigTabs();
  setupThermalLoadsBuilder(config);
  applyConfigFilters();
  updateConfigSystemOverviewBadges(config);
  const cfgContainer = document.getElementById("configuration-container");
  if (
    cfgContainer &&
    typeof cfgContainer.addEventListener === "function" &&
    !cfgContainer._boundOverviewBadges &&
    (!cfgContainer.dataset || !cfgContainer.dataset.boundOverviewBadges)
  ) {
    cfgContainer._boundOverviewBadges = true;
    if (cfgContainer.dataset) {
      cfgContainer.dataset.boundOverviewBadges = "1";
    }
    cfgContainer.addEventListener("change", () =>
      updateConfigSystemOverviewBadges(lastLoadedConfig)
    );
    cfgContainer.addEventListener("input", () =>
      updateConfigSystemOverviewBadges(lastLoadedConfig)
    );
  }
}

//build sections body, containing parameter/param containers (containing parameter/param inputs)
function buildParamContainers(
  section,
  section_parameters_definitions,
  config,
  header_input_list
) {
  //get the section container element
  let SectionContainer = document.getElementById(section);
  if (SectionContainer == null) {
    console.error("Unable to find Section container: " + section);
    return 0;
  }
  //get the body container inside the section (where the parameters will be appended)
  let SectionParamElement = SectionContainer.getElementsByClassName("section-body");
  if (SectionParamElement.length == 0) {
    console.error("Unable to find Section container or Section Body");
    return 0;
  }

  const invertersSection =
    section === "System" ? document.getElementById("Inverters") : null;
  const invertersBodies = invertersSection
    ? invertersSection.getElementsByClassName("section-body")
    : [];
  const invertersTargetBody =
    invertersBodies.length > 0 ? invertersBodies[0] : null;
  let routedToInverters = false;

  //loop though the sections parameters in definition file, generate and append param (div) elements for the section
  for (const [
    parameter_definition_name,
    parameter_definition_object,
  ] of Object.entries(section_parameters_definitions)) {
    //check parameter definitions have the required key values
    if (
      !("friendly_name" in parameter_definition_object) ||
      !("Description" in parameter_definition_object) ||
      !("input" in parameter_definition_object) ||
      !("default_value" in parameter_definition_object)
    ) {
      console.log(
        parameter_definition_name +
          " is missing some required values in the definitions file"
      );
      continue;
    }
    if (
      parameter_definition_object["input"] === "select" &&
      !("select_options" in parameter_definition_object)
    ) {
      console.log(
        parameter_definition_name +
          " is missing select_options values in the definitions file"
      );
      continue;
    }

    //check if param is set in the section header, if so skip building param
    if (header_input_list.includes(parameter_definition_name)) {
      continue;
    }

    // Generic array controls are excluded when another header owns cardinality.
    // Keep the original names in data attributes: a CSS class token cannot
    // preserve whitespace (or other non-class-token characters) in an ID.
    const has_array_buttons =
      parameter_definition_object["input"].search("array.") > -1 &&
      !(
        parameter_definition_object["input"] === "array.array.float" &&
        Array.isArray(parameter_definition_object["default_value"])
      ) &&
      !(section == "Deferrable Loads" && DEFERRABLE_ARRAY_PARAMS.includes(parameter_definition_name)) &&
      !(section == "Battery" && BATTERY_ARRAY_PARAMS.includes(parameter_definition_name)) &&
      !(section == "System" && INVERTER_ARRAY_PARAMS.includes(parameter_definition_name));

    // Definition metadata must be DOM text, not interpolated HTML. This also
    // avoids reparsing previously rendered siblings on each new parameter.
    const param_container = document.createElement("div");
    param_container.className = "param";
    param_container.id = parameter_definition_name;

    const heading = document.createElement("h5");
    heading.textContent = String(parameter_definition_object["friendly_name"]) + ":";
    const param_name = document.createElement("i");
    param_name.textContent = parameter_definition_name;
    param_container.append(heading, document.createTextNode(" "), param_name, document.createElement("br"));

    if (has_array_buttons) {
      const plus = document.createElement("button");
      plus.type = "button";
      plus.className = "input-plus";
      plus.dataset.paramName = parameter_definition_name;
      plus.textContent = "+";
      const minus = document.createElement("button");
      minus.type = "button";
      minus.className = "input-minus";
      minus.dataset.paramName = parameter_definition_name;
      minus.textContent = "-";
      param_container.append(plus, minus, document.createElement("br"));
    }

    const input_container = document.createElement("div");
    input_container.className = "param-input";
    // buildParamElement supplies only fixed markup and HTML-encodes all
    // configuration/default/option values. Metadata above never enters this
    // parser; it is written through textContent and DOM properties instead.
    const input_range = document.createRange();
    input_range.selectNodeContents(input_container);
    input_container.append(
      input_range.createContextualFragment(
        buildParamElement(parameter_definition_object, parameter_definition_name, config)
      )
    );
    param_container.append(input_container);

    const description = document.createElement("p");
    description.textContent = parameter_definition_object["Description"];
    param_container.append(description);

    const useInvertersCard =
      invertersTargetBody !== null &&
      INVERTER_CARD_PARAMS.has(parameter_definition_name);
    if (useInvertersCard) {
      routedToInverters = true;
    }
    const targetBody = useInvertersCard ? invertersTargetBody : SectionParamElement[0];
    targetBody.append(param_container);
  }

  const containerTargets =
    routedToInverters && invertersSection
      ? [SectionContainer, invertersSection]
      : [SectionContainer];

  //after looping though, build and appending the parameters in the corresponding section:
  //create add button (array plus) event listeners
  //plusElements() indexes its 2nd arg as param_definitions[section][name]; this
  //function only receives the single section's definitions, so wrap them back
  //under the section key. (The bare `param_definitions` used here previously is
  //not in this function's scope - the generic +/- path threw ReferenceError on
  //click; it was only ever reached by load_peak_hour_periods until #540 Part B
  //routed capacity_cost_per_kw / capacity_charge_interval_timesteps through it.)
  containerTargets.forEach(function (container) {
    let plus = container.querySelectorAll(".input-plus");
    plus.forEach(function (answer) {
      answer.addEventListener("click", () =>
        plusElements(
          answer.dataset.paramName,
          { [section]: section_parameters_definitions },
          section,
          {}
        )
      );
    });

    //create subtract button (array minus) event listeners
    let minus = container.querySelectorAll(".input-minus");
    minus.forEach(function (answer) {
      answer.addEventListener("click", () => minusElements(answer.dataset.paramName));
    });

    //check initial checkbox state, check "value" of input and match to "checked" value
    let checkbox = container.querySelectorAll("input[type='checkbox']");
    checkbox.forEach(function (answer) {
      let value = answer.value === "true";
      answer.checked = value;
    });
  });

  //loop though sections params again, check if param has a requirement, if so add a event listener to the required param input
  //if required param gets changed, trigger function to check if that required parameter matches the required value for the param
  //if false, add css class to param element to shadow it, to show that its unaccessible
  for (const [
    parameter_definition_name,
    parameter_definition_object,
  ] of Object.entries(section_parameters_definitions)) {
    //check if param has a requirement from definitions file
    if ("requires" in parameter_definition_object) {
      // get param requirement element
      const requirement_element = document.getElementById(
        Object.keys(parameter_definition_object["requires"])[0]
      );
      if (requirement_element == null) {
        console.debug(
          "unable to find " +
            Object.keys(parameter_definition_object["requires"])[0] +
            " param div container element"
        );
        continue;
      }

      // get param element that has requirement
      const param_element = document.getElementById(parameter_definition_name);
      if (param_element == null) {
        console.debug(
          "unable to find " +
            parameter_definition_name +
            " param div container element"
        );
        continue;
      }

      //obtain required param inputs, add event listeners
      let requirement_inputs =
        requirement_element.getElementsByClassName("param_input");
      //grab required value
      const requirement_value = Object.values(
        parameter_definition_object["requires"]
      )[0];

      //for all required inputs
      for (const input of requirement_inputs) {
        //if listener not already attached
        if (input.getAttribute("listener") !== "true") {
          //create event listener with arguments referencing the required param. param with requirement and required value
          input.addEventListener("input", () =>
            checkRequirements(input, param_element, requirement_value)
          );
          //manually run function to gain initial param element initial state
          checkRequirements(input, param_element, requirement_value);
        }
      }
    }
  }
}

//HTML-encode configuration/default/option values for use in buildParamElement's
//generated form controls. Definition metadata is assigned via DOM properties and
//textContent instead of being parsed as HTML (#1374).
//String() keeps the textual form template interpolation produced before.
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

//create html input element/s for a param container (called by buildParamContainers)
function buildParamElement(
  parameter_definition_object,
  parameter_definition_name,
  config
) {
  let type = "";
  let inputs = "";
  let type_specific_html = "";
  let type_specific_html_end = "";
  let placeholder = ""

  //switch statement to adjust generated html according to the parameter data type (definitions in definitions file)
  switch (parameter_definition_object["input"]) {
    case "array.int":
    //number
    case "int":
      type = "number";
      placeholder = Number.parseInt(parameter_definition_object["default_value"]);
      break;
    case "array.float":
    case "float":
      type = "number";
      placeholder = Number.parseFloat(parameter_definition_object["default_value"]);
      break;
    //text (string)
    case "array.string":
    case "string":
      type = "text";
      placeholder = parameter_definition_object["default_value"];
      break;
    case "array.time":
    //time ("00:00")
    case "time":
      type = "time";
      break;
    //checkbox (boolean)
    case "array.boolean":
    case "boolean":
      type = "checkbox";
      type_specific_html = `
              <label class="switch">
              `;
      type_specific_html_end = `
              <span class="slider"></span>
              </label>
              `;
      //#1116: default_value comes straight from the JSON definitions file, so a
      //declared boolean default (e.g. is_electric_load: true) parses as an
      //actual JS boolean, not the string "true" - comparing only against the
      //string left every declared-true default computing to false.
      placeholder =
        parameter_definition_object["default_value"] === true ||
        parameter_definition_object["default_value"] === "true";
      break;
    //selects (pick)
    case "select":
      //format selects later
      break;
    case "object":
    case "array.array.float":
      type = "text";
      placeholder = parameter_definition_object["default_value"] === null
        ? ""
        : JSON.stringify(parameter_definition_object["default_value"]);
      break;
  }

  //check default values saved in param definitions
  //definitions default value is used if none is found in the configs, or an array element has been added in the ui (deferrable load number increase or plus button pressed)
  //check if a param value is saved in the config file (if so overwrite definition default)
  let value = checkConfigParam(placeholder, config, parameter_definition_name);

  // A static numeric table (array.array.float with a list default, e.g. the
  // battery charge derating or the inverter power curves) uses one JSON field.
  // Per-load nested arrays declare a null default, may hold null entries and
  // keep their own per-load rendering and save paths.
  if (parameter_definition_object["input"] === "array.array.float" &&
      Array.isArray(parameter_definition_object["default_value"])) {
    const json = typeof value === "string" ? value : JSON.stringify(value ?? []);
    const escaped = json.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
    return `<input class="param_input" type="text" placeholder="[]" value="${escaped}">`;
  }
  // Object parameters (nested JSON such as heat_topology) get a multi-line
  // textarea: a one-line <input> is unusable for large nested structures.
  // Absent config, JSON null and the legacy string "null" all render as an
  // empty box. Content is entity-escaped; the browser decodes it back, so the
  // value round-trips through saveConfiguration's JSON.parse unchanged.
  if (parameter_definition_object["input"] === "object") {
    let json_text = "";
    if (value !== null && value !== undefined && value !== "" && value !== "null") {
      json_text = typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
    }
    json_text = json_text
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
    const placeholder_attr = String(placeholder ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll('"', "&quot;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
    return `<textarea class="param_input param-json" rows="10" spellcheck="false" placeholder="${placeholder_attr}">${json_text}</textarea>`;
  }

  //generate and return param input html,
  //check if param value is not an object, if so assume its a single value.
  if (typeof value !== "object") {
    //if select, generate and return select elements instead of input
    if (parameter_definition_object["input"] == "select") {
      let inputs = `<select class="param_input">`;
      for (const options of parameter_definition_object["select_options"]) {
        let selected = ""
        //if item in select is the same as the config value, then append "selected" tag
        if (options==value) {selected = `selected="selected"`}
        //explicit value: without it the browser derives the value from the option text,
        //stripping and collapsing whitespace, so saveConfiguration would read it altered
        inputs += `<option ${selected} value="${escapeHtml(options)}">${escapeHtml(options)}</option>`;
      }
      inputs += `</select>`;
      return inputs;
    }
    // generate param input html and return
    else {
      //#1116: set the checked attribute directly at render time - dynamically
      //added rows (plusElements) never run buildParamContainers' one-time
      //checkbox normalisation pass, so a checkbox created there must already
      //reflect its true/false value without relying on that later pass.
      let checked_attr = type === "checkbox" && value === true ? "checked" : "";
      return `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder="${escapeHtml(parameter_definition_object["default_value"])}" value="${escapeHtml(value)}" ${checked_attr}>
          ${type_specific_html_end}
          `;
    }
  }
  // else if object, loop though array of values, generate input element per value, and and return
  else {
    // null default: render a single empty input so the section keeps rendering
    if (value === null) {
      return `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder="${escapeHtml(placeholder)}" value="">
          ${type_specific_html_end}
          `;
    }
    //for items such as load_peak_hour_periods (object of objects with arrays)
    // exclude null: typeof null === "object" is a JS gotcha — null elements must fall to the array branch
    if (typeof Object.values(value)[0] === "object" && Object.values(value)[0] !== null) {
      for (let param of Object.values(value)) {
        for (let items of Object.values(param)) {
          inputs += `<input class="param_input" type="${type}" placeholder="${escapeHtml(Object.values(items)[0])}" value="${escapeHtml(
            Object.values(items)[0]
          )}">`;
        }
        inputs += `</br>`;
      }
      return inputs;
    }
    // array of values
    else {
      let inputs = "";
      for (let param of value) {
        //#1116: same as the single-value branch above - set checked directly
        //so a freshly-created array.boolean row (e.g. adding a deferrable
        //load) starts in the correct state without a separate normalisation pass.
        let checked_attr = type === "checkbox" && param === true ? "checked" : "";
        inputs += `
          ${type_specific_html}
          <input class="param_input" type="${type}" placeholder="${escapeHtml(parameter_definition_object["default_value"])}" value="${escapeHtml(param)}" ${checked_attr}>
          ${type_specific_html_end}
          `;
      }
      return inputs;
    }
  }
}

//add param inputs in param div container (for type array)
function plusElements(
  parameter_definition_name,
  param_definitions,
  section,
  config
) {
  let param_element = document.getElementById(parameter_definition_name);
  if (param_element == null) {
    console.log(
      "Unable to find " + parameter_definition_name + " param div container"
    );
    return 1;
  }
  let param_input_container =
    param_element.getElementsByClassName("param-input")[0];
  // Append a new input WITHOUT re-serialising the existing ones. `innerHTML +=`
  // would round-trip every sibling through its HTML string form, which only
  // carries the `value` attribute - any value the user has typed but not saved
  // lives on the input's `.value` property and would be reset (e.g. entering
  // 3, 7 then removing and re-adding the 2nd row blanked the first back to its
  // default). insertAdjacentHTML parses and inserts only the new node.
  param_input_container.insertAdjacentHTML(
    "beforeend",
    buildParamElement(
      param_definitions[section][parameter_definition_name],
      parameter_definition_name,
      config
    )
  );
}

//Remove param inputs in param div container (minimum `minimum_inputs`, default 1).
//#1116: the deferrable-loads shrink path passes 0 so number_of_deferrable_loads
//can reach 0 (EMHASS accepts num_def_loads=0); every other caller (generic
//+/-, number_of_batteries) omits the argument and keeps the prior minimum-1
//floor unchanged.
function minusElements(param, minimum_inputs = 1) {
  let param_element = document.getElementById(param);
  let param_input
  if (param_element == null) {
    console.log(
      "Unable to find " + param + " param div container"
    );
    return 1;
  }
  let param_input_list = param_element.getElementsByTagName("input");
  if (param_input_list.length == 0) {
    console.log(
      "Unable to find " + param + " param input/s"
    );
    return 1;
  }

  //verify if input is a boolean (if so remove parent slider/switch element with input)
  if (
    param_input_list[param_input_list.length - 1].parentNode.tagName === "LABEL"
  ) {
    param_input = param_input_list[param_input_list.length - 1].parentNode;
  } else {
    param_input = param_input_list[param_input_list.length - 1];
  }

  //if param is "load_peak_hour_periods", remove both start and end param inputs as well as the line brake tag separating the inputs
  if (param == "load_peak_hour_periods") {
    if (param_input_list.length > 2) {
      let brs = document.getElementById(param).getElementsByTagName("br");
      param_input_list[param_input_list.length - 1].remove();
      param_input_list[param_input_list.length - 1].remove();
      brs[brs.length - 1].remove();
    }
  } else if (param_input_list.length > minimum_inputs) {
    param_input.remove();
  }
}

//check requirement_element inputs,
//if requirement_element don't match requirement_value, add .requirement-disable class to param_element
//else remove class
function checkRequirements(
  requirement_element,
  param_element,
  requirement_value
) {
  let requirement_element_value
  //get current value of required element
  if (requirement_element.type == "checkbox") {
    requirement_element_value = requirement_element.checked;
  } else {
    requirement_element_value = requirement_element.value;
  }

  if (requirement_element_value != requirement_value) {
    if (!param_element.classList.contains("requirement-disable")) {
      param_element.classList.add("requirement-disable");
    }
  } else if (param_element.classList.contains("requirement-disable")) {
      param_element.classList.remove("requirement-disable");
  }
}

//on header input change, execute accordingly
function headerElement(element, param_definitions, config) {
  //obtain section body element
  let section_card = element.closest(".section-card");
  let param_list 
  let difference
  if (section_card == null) {
    console.log("Unable to obtain section-card");
    return 1;
  }
  let param_container = section_card.getElementsByClassName("section-body");
  if (param_container.length > 0) {
    param_container = section_card.getElementsByClassName("section-body")[0];
  } else {
    console.log("Unable to obtain section-body");
    return 1;
  }

  switch (element.id) {
    //if set_use_battery, add or remove battery section (inc. params)
    case "set_use_battery":
      if (element.checked) {
        param_container.innerHTML = "";
        //#610: also exclude number_of_batteries here (its own header case below
        //owns it) - otherwise this rebuild would render a SECOND, duplicate
        //"number_of_batteries" .param row inside the section body alongside the
        //real header input, since this rebuild's own exclusion list is separate
        //from the outer header_input_list used at initial page load.
        buildParamContainers("Battery", param_definitions["Battery"], config, [
          "set_use_battery",
          "number_of_batteries",
        ]);
        element.checked = true;
        setupDynamicVisibility();
      } else {
        param_container.innerHTML = "";
        applyConfigFilters();
      }
      updateConfigSystemOverviewBadges(lastLoadedConfig || config);
      break;

    //if set_use_pv, add or remove PV section (inc. related params)
    case "set_use_pv":
      if (element.checked) {
        param_container.innerHTML = "";
        buildParamContainers("Solar System (PV)", param_definitions["Solar System (PV)"], config, [
          "set_use_pv",
        ]);
        element.checked = true;
        setupDynamicVisibility();
      } else {
        param_container.innerHTML = "";
        applyConfigFilters();
      }
      updateConfigSystemOverviewBadges(lastLoadedConfig || config);
      break;

    //if number_of_deferrable_loads, the number of inputs in the "Deferrable Loads" section should add up to number_of_deferrable_loads value in header
    //#1116: only the genuine per-load array params (DEFERRABLE_ARRAY_PARAMS) are
    //grown/shrunk here - not every ".param" in the section, since heat_topology
    //is a single object param whose cardinality is not tied to
    //number_of_deferrable_loads (mirrors the number_of_batteries case below).
    case "number_of_deferrable_loads":
      //get a list of only the deferrable-load array params in section
      param_list = Array.from(
        param_container.getElementsByClassName("param")
      ).filter((p) => DEFERRABLE_ARRAY_PARAMS.includes(p.id));
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_deferrable_loads"
        );
        return 1;
      }
      //calculate how much off the first deferrable-load array param's input elements amount to is, comparing to the number_of_deferrable_loads value
      difference =
        Number.parseInt(element.value) -
        param_list[0].querySelectorAll("input").length;
      //add elements based on how many elements are missing
      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            //append element, do not pass config to obtain default parameter from definitions file
            plusElements(param.id, param_definitions, "Deferrable Loads", {});
          }
        }
      }
      //subtract elements based how many elements its over
      //#1116: pass minimum 0 - number_of_deferrable_loads may reach 0 (EMHASS
      //accepts num_def_loads=0), unlike the generic/battery paths below which
      //keep the default minimum-1 floor.
      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id, 0);
          }
        }
      }
      break;

    //#610: the 15 per-battery array params
    //(BATTERY_ARRAY_PARAMS) in the "Battery" section should add up to
    //number_of_batteries, mirroring the number_of_deferrable_loads case above.
    //Unlike "Deferrable Loads", "Battery" also holds global scalar flags
    //(set_use_battery, set_nocharge_from_grid, set_battery_dynamic, ...) that
    //must NOT be grown/shrunk by this count, so param_list is filtered down to
    //BATTERY_ARRAY_PARAMS rather than every ".param" in the section.
    case "number_of_batteries":
      //get a list of only the battery array params in section
      param_list = Array.from(
        param_container.getElementsByClassName("param")
      ).filter((p) => BATTERY_ARRAY_PARAMS.includes(p.id));
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_batteries"
        );
        return 1;
      }
      //calculate how much off the first battery array param's input elements amount to is, comparing to the number_of_batteries value
      difference =
        Number.parseInt(element.value) -
        param_list[0].querySelectorAll("input").length;
      //add elements based on how many elements are missing
      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            //append element, do not pass config to obtain default parameter from definitions file
            plusElements(param.id, param_definitions, "Battery", {});
          }
        }
      }
      //subtract elements based how many elements its over
      if (difference < 0) {
        for (let i = difference; i <= -1; i++) {
          for (const param of param_list) {
            minusElements(param.id);
          }
        }
      }
      break;
  }
}

//#1150: number_of_inverters as a whole number of at least 1, else null (an
//emptied, fractional or partly typed count)
//(Number() reads "", " " and null as 0, which the check below refuses)
function parseInverterCount(raw) {
  const count = Number(raw);
  return Number.isInteger(count) && count >= 1 ? count : null;
}

//#1150: the first input of a param row, or undefined when the row is missing
function paramInputOf(param_id) {
  const el = document.getElementById(param_id);
  if (!el || typeof el.getElementsByClassName !== "function") {
    return undefined;
  }
  return el.getElementsByClassName("param_input")[0];
}

//#1150: keep the INVERTER_ARRAY_PARAMS at number_of_inverters entries, the way
//the "number_of_batteries" header case keeps the battery lists at
//number_of_batteries. number_of_inverters stays a normal row with its label and
//help in the System section, with the lists it sizes, so this listens on that
//row's input rather than on a section header input.
function setupInverterSync(param_definitions, config) {
  const count_input = paramInputOf("number_of_inverters");
  if (count_input == null) {
    return;
  }
  //first render: shape the lists from the stored configuration
  syncInverterLists(count_input.value, param_definitions, config);
  updateInverterCurveAccess(count_input.value, param_definitions);
  //"change" rather than "input": typing 12 would otherwise pass through 1 on the
  //way and drop what was typed in the rows from the second inverter on
  count_input.addEventListener("change", () => {
    syncInverterLists(count_input.value, param_definitions, null);
    updateInverterCurveAccess(count_input.value, param_definitions);
  });
  //buildParamContainers attached the generic "requires" listener to this input
  //before this runs, and listeners fire in the order they were added, so this
  //one has the last word on the curves
  const hybrid_input = paramInputOf("inverter_is_hybrid");
  if (hybrid_input != null) {
    hybrid_input.addEventListener("input", () =>
      updateInverterCurveAccess(count_input.value, param_definitions)
    );
  }
}

//#1150: the alert box text of the last alert syncInverterLists raised, so a
//later sync that works can take it down again (and never anyone else's alert)
let inverter_sync_alert = null;

//#1150: grow or shrink each INVERTER_ARRAY_PARAMS list to the count. Each list
//is counted on its own, so lists of different lengths all end at the count.
//`config` is the stored configuration on the first render and null after a
//change of the count. The first render must not change what a stored
//configuration means, so it only grows a list from a single row (a scalar, a
//one-entry list or a missing key), the way the backend reads a single value.
//Any other list whose length differs from the count stays as stored, an alert
//names it, and the save refuses it by name as it does without this sync.
function syncInverterLists(raw_count, param_definitions, config) {
  const count = parseInverterCount(raw_count);
  if (count === null) {
    //leave the lists alone: the save reports a count that is not usable
    return;
  }
  if (count > INVERTER_SYNC_LIMIT) {
    showInverterSyncAlert(
      `number_of_inverters is ${String(raw_count).trim()}: the per-inverter lists follow it only up to ${INVERTER_SYNC_LIMIT} inverters, so they were left as they are.`
    );
    return;
  }
  const first_render = config != null;
  const mismatched = [];
  for (const name of INVERTER_ARRAY_PARAMS) {
    const param_element = document.getElementById(name);
    if (param_element == null) {
      continue;
    }
    let rows = param_element.getElementsByClassName("param_input").length;
    if (first_render && rows !== count && rows !== 1) {
      mismatched.push(`${name} (${rows} entries)`);
      continue;
    }
    const fill = inverterNewRowFill(name, first_render ? config : null);
    for (; rows < count; rows++) {
      plusElements(name, param_definitions, "System", fill);
    }
    for (; rows > count; rows--) {
      minusElements(name);
    }
  }
  if (mismatched.length > 0) {
    showInverterSyncAlert(
      `number_of_inverters is ${count} but the stored configuration has ${mismatched.join(", ")}. They are shown as stored and the save will refuse them until each has ${count}. To resize them, change number_of_inverters to another value and back.`
    );
  } else {
    clearInverterSyncAlert();
  }
}

//#1150: what a new row holds, as the `config` argument of plusElements. A cap
//row starts empty: above one inverter the backend refuses a cap that is just
//the 5000 W default or a copy of a one-inverter site total, so the save asks
//for each inverter's cap instead of storing one of those. A broadcast row takes
//the definition default after a change of the count, as new battery rows do. On
//the first render (`config` set) a broadcast param stored as one value repeats
//it, since that value means every inverter to the backend; a key missing from
//the stored configuration keeps the definition default, which the backend
//broadcasts too.
function inverterNewRowFill(name, config) {
  if (INVERTER_CAP_PARAMS.includes(name)) {
    return { [name]: null };
  }
  if (config == null || !(name in config)) {
    return {};
  }
  const stored = config[name];
  return { [name]: Array.isArray(stored) ? stored[0] : stored };
}

function showInverterSyncAlert(text) {
  errorAlert(text);
  inverter_sync_alert = document.getElementById("alert-text")?.textContent ?? null;
}

function clearInverterSyncAlert() {
  //an alert box showing any other text belongs to someone else and stays
  if (
    inverter_sync_alert !== null &&
    document.getElementById("alert-text")?.textContent === inverter_sync_alert
  ) {
    document.getElementById("alert").style.display = "none";
  }
  inverter_sync_alert = null;
}

//#1150: above one inverter the power curves are used whatever inverter_is_hybrid
//says (the backend does not read it there), so they stay editable; at one
//inverter, or with a count that is not usable, they keep the "requires" rule
//from the definitions file
function updateInverterCurveAccess(raw_count, param_definitions) {
  const count = parseInverterCount(raw_count);
  for (const name of INVERTER_CURVE_PARAMS) {
    const param_element = document.getElementById(name);
    const requires = param_definitions?.["System"]?.[name]?.["requires"];
    if (param_element == null || requires == null) {
      continue;
    }
    const requirement_input = paramInputOf(Object.keys(requires)[0]);
    if (count !== null && count > 1) {
      param_element.classList.remove("requirement-disable");
    } else if (requirement_input != null) {
      checkRequirements(requirement_input, param_element, Object.values(requires)[0]);
    }
  }
}

//checks parameter value in config, updates value if exists
function checkConfigParam(value, config, parameter_definition_name) {
  if (config !== null && config !== undefined) {
    //check if parameter has a saved value
    if (parameter_definition_name in config) {
      value = config[parameter_definition_name];
    }
  }
  return value;
}

//send all parameter input values to EMHASS, to save to config.json and param.pkl
async function saveConfiguration(param_definitions) {
  //start wth none
  let config = {};
  let param_inputs
  let param_element

  //if section-cards (config sections/list) exists
  let config_card = document.getElementsByClassName("section-card");
  //check if page is in list or box view
  let config_box_element = document.getElementById("config-box");

  //if true, in list view
  if (Boolean(config_card.length)) {
    //retrieve params and their input/s by looping though param_definitions list
    //loop through the sections
    for (const [section, section_object] of Object.entries(
      param_definitions
    )) {
      //loop through parameters
      for (let [
        parameter_definition_name,
        parameter_definition_object,
      ] of Object.entries(section_object)) {
        let param_values = []; //stores the obtained param input values
        let param_array = false;
        //get param container
        param_element = document.getElementById(parameter_definition_name);
        if (param_element == null) {
          console.debug(
            "unable to find " +
              parameter_definition_name +
              " param div container element, skipping this param"
          );
        }
        //extract input/s and their value/s from param container div
        else {
          if (param_element.tagName !== "INPUT") {
            param_inputs = param_element.getElementsByClassName("param_input");
          } else {
            //check if param_element is also param_input (ex. for header parameters)
            param_inputs = [param_element];
          }

          // loop though param_inputs, extract the element/s values
          for (let input of param_inputs) {
            switch (input.type) {
              case "number":
                param_values.push(Number.parseFloat(input.value));
                break;
              case "checkbox":
                param_values.push(input.checked);
                break;
              default:
                param_values.push(input.value);
                break;
            }
          }
          //obtain param input type from param_definitions, check if param should be formatted as an array
          param_array = Boolean(
            !parameter_definition_object["input"].search("array")
          );

          //build parameters using values extracted from param_inputs

          // Static numeric tables must be saved as JSON, not flattened strings.
          // Only the structure is checked here; the physics of each table is
          // validated authoritatively in Python.
          if (parameter_definition_object["input"] === "array.array.float" &&
              Array.isArray(parameter_definition_object["default_value"])) {
            try {
              const value = JSON.parse((param_values[0] ?? "").trim() || "[]");
              const numericRow = (row) => Array.isArray(row) &&
                row.every((item) => typeof item === "number" && Number.isFinite(item));
              const numericTable = (table) => Array.isArray(table) && table.every(numericRow);
              if (!numericTable(value) &&
                  !(Array.isArray(value) && value.every(numericTable))) {
                throw new Error("Expected nested numeric arrays");
              }
              config[parameter_definition_name] = value;
            } catch (_) {
              errorAlert(parameter_definition_name + ": enter a JSON array of numeric rows or tables, for example [[0.5, 0.84]].");
              return 0;
            }
            continue;
          }

          // object-type: JSON.parse the textarea value; treat "" and "null" as
          // JSON null. Trim so stray whitespace/newlines in an otherwise empty
          // textarea do not read as invalid JSON.
          if (parameter_definition_object["input"] === "object") {
            const raw = (param_values[0] ?? "").toString().trim();
            if (raw === "" || raw === "null") {
              config[parameter_definition_name] = null;
            } else {
              try {
                config[parameter_definition_name] = JSON.parse(raw);
              } catch (_) {
                errorAlert(parameter_definition_name + ": invalid JSON — please check the value and try again.");
                return 0;
              }
            }
            continue;
          }

          // If time with 2 sets (load_peak_hour_periods)
          if (
            parameter_definition_object["input"] == "array.time" &&
            param_values.length % 2 === 0
          ) {
            config[parameter_definition_name] = {};
            for (let i = 0; i < param_values.length; i++) {
              config[parameter_definition_name][
                "period_hp_" +
                  (Object.keys(config[parameter_definition_name]).length + 1)
              ] = [{ start: param_values[i] }, { end: param_values[++i] }];
            }
            continue;
          }

          //single value
          if (param_values.length && !param_array) {
            config[parameter_definition_name] = param_values[0];
          }

          //array value
          else if (param_values.length) {
            config[parameter_definition_name] = param_values;
          }
          //#1116: Deferrable Loads may intentionally render zero managed rows
          //when number_of_deferrable_loads=0. Submit those owned fields as []
          //without changing zero-input save semantics for unrelated arrays.
          else if (
            param_array &&
            section === "Deferrable Loads" &&
            DEFERRABLE_ARRAY_PARAMS.includes(parameter_definition_name)
          ) {
            config[parameter_definition_name] = [];
          }
        }
      }
    }
    const builtThermalList =
      typeof buildThermalLoadsConfig === "function"
        ? buildThermalLoadsConfig()
        : null;
    if (builtThermalList !== null) {
      const hasThermalModel = builtThermalList.some(
        (entry) => entry && Object.keys(entry).length > 0
      );
      const hadExistingThermal =
        typeof lastLoadedConfig !== "undefined" &&
        lastLoadedConfig &&
        Array.isArray(lastLoadedConfig.def_load_config);
      if (hasThermalModel || hadExistingThermal) {
        config["def_load_config"] = builtThermalList;
      }
    }
  }

  //if box view, extract json from box view
  else if (config_box_element !== null) {
    //try and parse json from box
    try {
      config = JSON.parse(config_box_element.value);
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
  // else, cant find box or list view
  else {
    errorAlert("There has been an error verifying box or list view");
  }

  //finally, send built config to emhass
  const timeout = AbortSignal.timeout?.(REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`set-config`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(config),
      signal: timeout,
    });
    const saved = response.status === 200 || response.status === 201;
    showChangeStatus(response.status, saved ? [] : await readErrorLog(response));
  } catch (error) {
    //a request that timed out may still have been applied by the server (older Chrome aborts with an
    //AbortError rather than a TimeoutError, so ask the signal)
    const outcome = timeout?.aborted ? " The server may still have saved it." : "";
    showChangeStatus(0, [`Unable to save the configuration: ${error.message}.${outcome} Reload the page to check the stored settings.`]);
  }
}

//Toggle between box (json) and list view
async function ToggleView(param_definitions, list_html, default_reset) {
  let selected = "";
  config = {};

  //find out if list or box view is active
  let configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //get yaml button
  let yaml_button = document.getElementById("yaml");
  if (yaml_button == null) {
    console.log("Unable to obtain yaml button");
  }

  // if section-cards (config sections/list) exists
  let config_card = configuration_container.getElementsByClassName("section-card");
  //selected view (0 = box)
  let selected_view = Boolean(config_card.length);

  //if default_reset is passed do not switch views, instead reinitialize view with default config as values
  if (default_reset) {
    selected_view = !selected_view;
    //obtain default config as config (when pressing the default button)
    config = await ObtainDefaultConfig();
  } else {
    //obtain latest config
    config = await obtainConfig();
  }
  //keep the current view if the configuration could not be obtained
  if (config === null) {
    return;
  }

  //if array is empty assume json box is selected
  if (selected_view) {
    selected = "list";
  } else {
    selected = "box";
  }
  //remove contents of current view
  configuration_container.innerHTML = "";
  //build new view
  const filterBar =
    typeof document.querySelector === "function"
      ? document.querySelector(".config-filter-bar")
      : null;
  switch (selected) {
    case "box":
      //load list
      loadConfigurationListView(param_definitions, config, list_html);
      yaml_button.style.display = "none";
      if (filterBar) {
        filterBar.style.display = "";
      }
      break;
    case "list":
      //load box
      loadConfigurationBoxPage(config);
      yaml_button.style.display = "block";
      if (filterBar) {
        filterBar.style.display = "none";
      }
      break;
  }
}

//load box (json textarea) view
async function loadConfigurationBoxPage(config) {
  //get configuration container element
  let configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //append configuration container with textbox area
  configuration_container.innerHTML = `
      <textarea id="config-box" rows="30" placeholder="{}" aria-label="Configuration JSON or YAML text editor"></textarea>
      `;
  //set created textarea box with retrieved config
  //.value, not innerHTML: a textarea's HTML parsing would decode &amp; etc. in the data (#1290)
  const box = document.getElementById("config-box");
  box.value = JSON.stringify(config, null, 2);
  lastLoadedConfig = config;
  updateConfigSystemOverviewBadges(config);
  if (typeof box.addEventListener === "function") {
    box.addEventListener("input", () => {
      try {
        const parsed = JSON.parse(box.value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
          lastLoadedConfig = parsed;
          updateConfigSystemOverviewBadges(parsed);
        }
      } catch (_err) {
        // ignore incomplete JSON/YAML while editing
      }
    });
  }
}

//function in control of status icons and alert box from a fetch request
async function showChangeStatus(status, logJson) {
  let loading = document.getElementById("loader"); //element showing statuses
  if (loading === null) {
    console.log("unable to find loader element");
    return 1;
  }
  if (status === 200 || status === 201) {
    //if status is 200 or 201, then show a tick
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=tick>&#x2713;</p>`;
    //a request that succeeds ends the failure shown for the one before it
    document.getElementById("alert").style.display = "none";
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
  //remove tick/cross after some time
  setTimeout(() => {
    loading.innerHTML = "";
  }, 4000);
}

//simple function to write text to the alert box
async function errorAlert(text) {
  if (
    document.getElementById("alert-text") !== null &&
    document.getElementById("alert") !== null
  ) {
    document.getElementById("alert-text").textContent = "\r\n" + text + "\r\n";
    document.getElementById("alert").style.display = "block";
    document.getElementById("alert").style.textAlign = "left";
  }
  return 0;
}

//convert yaml box into json box
async function yamlToJson() {
  //get box element
  let config_box_element = document.getElementById("config-box");
  if (config_box_element == null) {
    errorAlert("Unable to obtain config box");
  } else {
    const response = await fetch(`get-json`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: config_box_element.value,
    });
    let response_status = response.status; //return status
    if (response_status == 200 || response_status == 201) {
      showChangeStatus(response_status, {});
      let blob = await response.blob(); //get data blob
      config = await new Response(blob).json(); //obtain json from blob
      config_box_element.value = JSON.stringify(config, null, 2);
    } else {
      showChangeStatus(response_status, await response.json());
    }
  }
  return 0;
}

const BASIC_PARAM_IDS = new Set([
  "costfun",
  "optimization_time_step",
  "historic_days_to_retrieve",
  "delta_forecast_daily",
  "load_forecast_method",
  "sensor_power_photovoltaics",
  "sensor_power_load_no_var_loads",
  "load_cost_forecast_method",
  "load_peak_hour_periods",
  "load_peak_hours_cost",
  "load_offpeak_hours_cost",
  "production_price_forecast_method",
  "photovoltaic_production_sell_price",
  "capacity_cost_per_kw",
  "capacity_charge_interval_timesteps",
  "set_use_pv",
  "weather_forecast_method",
  "pv_module_model",
  "pv_inverter_model",
  "surface_tilt",
  "surface_azimuth",
  "modules_per_string",
  "strings_per_inverter",
  "inverter_ac_output_max",
  "maximum_power_from_grid",
  "maximum_power_to_grid",
  "set_use_battery",
  "number_of_batteries",
  "battery_nominal_energy_capacity",
  "battery_discharge_power_max",
  "battery_charge_power_max",
  "battery_minimum_state_of_charge",
  "battery_maximum_state_of_charge",
  "battery_target_state_of_charge",
  "number_of_deferrable_loads",
  "nominal_power_of_deferrable_loads",
  "operating_hours_of_each_deferrable_load",
  "start_timesteps_of_each_deferrable_load",
  "end_timesteps_of_each_deferrable_load",
  "continual_publish",
  "logging_level",
]);

let activeConfigTab = "all";

function initConfigTabs() {
  if (typeof document.querySelectorAll !== "function") {
    return;
  }
  const buttons = Array.from(document.querySelectorAll(".config-tab-btn"));
  const cards = Array.from(document.querySelectorAll(".section-card"));
  if (!buttons.length || !cards.length) {
    return;
  }

  function selectTab(tabId) {
    activeConfigTab = tabId;
    buttons.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset?.configTab === tabId);
    });
    cards.forEach((card) => {
      card.style.display = tabId === "all" || card.id === tabId ? "" : "none";
    });
  }

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      selectTab(btn.dataset?.configTab || "all");
      applyConfigFilters();
    });
  });

  selectTab(activeConfigTab);
}

function resolveControllerInput(controllerId) {
  const controllerDiv = document.getElementById(controllerId);
  if (!controllerDiv) {
    return null;
  }
  if (controllerDiv.tagName === "INPUT" || controllerDiv.tagName === "SELECT") {
    return controllerDiv;
  }
  return typeof controllerDiv.querySelector === "function"
    ? controllerDiv.querySelector("input, select")
    : null;
}

function setupDynamicVisibility() {
  const rules = [
    {
      controller: "use_influxdb",
      targets: [
        "influxdb_host",
        "influxdb_port",
        "influxdb_username",
        "influxdb_password",
        "influxdb_database",
        "influxdb_measurement",
        "influxdb_retention_policy",
        "influxdb_use_ssl",
        "influxdb_verify_ssl",
      ],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "use_victoriametrics",
      targets: [
        "victoriametrics_host",
        "victoriametrics_port",
        "victoriametrics_database",
        "victoriametrics_metric_regex",
        "victoriametrics_entity_label",
        "victoriametrics_use_ssl",
        "victoriametrics_verify_ssl",
      ],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "load_forecast_method",
      targets: [
        "mlforecaster_weather_features",
        "model_type",
        "var_model",
        "sklearn_model",
        "regression_model",
        "num_lags",
        "split_date_delta",
        "n_trials",
        "perform_backtest",
      ],
      when: (el) => el.value === "mlforecaster",
    },
    {
      controller: "optimization_backend",
      targets: ["site"],
      when: (el) => el.value === "dantzig_wolfe",
    },
    {
      controller: "load_cost_forecast_method",
      targets: [
        "load_peak_hour_periods",
        "tariff_schedule_time_zone",
        "load_peak_hours_cost",
        "load_offpeak_hours_cost",
      ],
      when: (el) => el.value === "hp_hc_periods",
    },
    {
      controller: "production_price_forecast_method",
      targets: ["photovoltaic_production_sell_price"],
      when: (el) => el.value === "constant",
    },
    {
      controller: "set_use_adjusted_pv",
      targets: [
        "adjusted_pv_regression_model",
        "adjusted_pv_solar_elevation_threshold",
        "adjusted_pv_model_max_age",
      ],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "weather_forecast_method",
      targets: ["open_meteo_cache_max_age"],
      when: (el) => el.value === "open-meteo",
    },
    {
      controller: "weather_forecast_method",
      targets: ["weather_forecast_pv_quantile_bias"],
      when: (el) => el.value === "solcast",
    },
    {
      controller: "set_use_battery_identification",
      targets: [
        "battery_identification_trust_tier",
        "battery_identification_model_max_age",
      ],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "set_battery_dynamic",
      targets: ["battery_dynamic_max", "battery_dynamic_min"],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "cop_solver",
      targets: ["cop_solver_tolerance", "cop_hx_approach"],
      when: (el) => el.value !== "static",
    },
    {
      controller: "inverter_is_hybrid",
      targets: ["inverter_ac_input_max", "inverter_efficiency_ac_dc"],
      when: (el) => Boolean(el.checked),
    },
    {
      controller: "inverter_is_hybrid",
      targets: ["inverter_power_curve_ac_dc"],
      when: (el) => {
        const count = parseInverterCount(paramInputOf("number_of_inverters")?.value);
        return Boolean(el.checked) || (count !== null && count > 1);
      },
    },
    {
      controller: "number_of_inverters",
      targets: ["inverter_power_curve_ac_dc"],
      when: (el) => {
        const count = parseInverterCount(el.value);
        const hybrid = Boolean(paramInputOf("inverter_is_hybrid")?.checked);
        return hybrid || (count !== null && count > 1);
      },
    },
  ];

  function evaluateAllRules() {
    const targetVisibility = new Map();
    rules.forEach((rule) => {
      const ctrl = resolveControllerInput(rule.controller);
      if (!ctrl) {
        return;
      }
      const visible = rule.when(ctrl);
      rule.targets.forEach((targetId) => {
        // Track whether at least one matching rule says visible, vs hidden
        if (!targetVisibility.has(targetId)) {
          targetVisibility.set(targetId, visible);
        } else if (visible) {
          targetVisibility.set(targetId, true);
        }
      });
    });

    targetVisibility.forEach((visible, targetId) => {
      const el = document.getElementById(targetId);
      if (!el || !el.dataset) {
        return;
      }
      el.dataset.dynamicHidden = visible ? "0" : "1";
    });
    applyConfigFilters();
  }

  const boundControllers = new Set();
  rules.forEach((rule) => {
    if (boundControllers.has(rule.controller)) {
      return;
    }
    const ctrl = resolveControllerInput(rule.controller);
    if (!ctrl) {
      return;
    }
    boundControllers.add(rule.controller);
    const alreadyBound =
      Boolean(ctrl._boundDynVis) ||
      ctrl.dataset?.boundDynVis === "1";
    if (!alreadyBound && typeof ctrl.addEventListener === "function") {
      ctrl._boundDynVis = true;
      if (ctrl.dataset) {
        ctrl.dataset.boundDynVis = "1";
      }
      ctrl.addEventListener("change", evaluateAllRules);
      ctrl.addEventListener("input", evaluateAllRules);
    }
  });

  evaluateAllRules();
}

function applyConfigFilters() {
  if (typeof document.querySelectorAll !== "function") {
    return;
  }
  const searchEl = document.getElementById("config-search-input");
  const levelEl = document.getElementById("config-level-filter");
  const query = searchEl ? searchEl.value.trim().toLowerCase() : "";
  const level = levelEl ? levelEl.value : "all";

  const params = Array.from(document.querySelectorAll(".section-card .param"));
  params.forEach((paramDiv) => {
    const paramId = paramDiv.id || "";
    const text = paramDiv.textContent ? paramDiv.textContent.toLowerCase() : "";
    const dynHidden = paramDiv.dataset?.dynamicHidden === "1";
    const matchQuery = !query || paramId.toLowerCase().includes(query) || text.includes(query);
    const matchLevel = level === "all" || BASIC_PARAM_IDS.has(paramId);

    // Explicit search surfaces matching parameters even if collapsed by a dynamic toggle
    const show = query ? matchQuery && matchLevel : !dynHidden && matchLevel;
    paramDiv.style.display = show ? "" : "none";
  });

  const cards = Array.from(document.querySelectorAll(".section-card"));
  cards.forEach((card) => {
    if (activeConfigTab !== "all" && card.id !== activeConfigTab && !query) {
      card.style.display = "none";
      return;
    }
    if (!query) {
      card.style.display = activeConfigTab === "all" || card.id === activeConfigTab ? "" : "none";
      return;
    }
    const visibleParams = Array.from(card.querySelectorAll(".param")).some(
      (p) => p.style.display !== "none"
    );
    card.style.display = visibleParams ? "" : "none";
  });
}

let thermalLoadStates = [];

function parseTempArrayText(raw) {
  if (Array.isArray(raw)) {
    return raw.map((x) => {
      if (x === null || x === undefined || x === "") {
        return null;
      }
      const n = Number(x);
      return Number.isFinite(n) ? n : null;
    });
  }
  const str = String(raw || "").trim();
  if (!str) {
    return [];
  }
  if (str.startsWith("[")) {
    try {
      const parsed = JSON.parse(str);
      if (Array.isArray(parsed)) {
        return parsed.map((x) => {
          if (x === null || x === undefined || x === "") {
            return null;
          }
          const n = Number(x);
          return Number.isFinite(n) ? n : null;
        });
      }
    } catch (_err) {
      // fall through to CSV split
    }
  }
  const tokens = [];
  for (const chunk of str.split(/[,;]/)) {
    const trimmed = chunk.trim();
    if (!trimmed) {
      tokens.push("");
    } else if (/\s/.test(trimmed)) {
      tokens.push(...trimmed.split(/\s+/));
    } else {
      tokens.push(trimmed);
    }
  }
  return tokens.map((token) => {
    if (!token || token.toLowerCase() === "null" || token.toLowerCase() === "none") {
      return null;
    }
    const n = Number(token);
    return Number.isFinite(n) ? n : null;
  });
}

// 7 days at 1-minute resolution: upper guard against accidental huge allocations
const MAX_THERMAL_SCHEDULE_STEPS = 7 * 24 * 60;

function getLocalHourInTimeZone(dateObj, timeZone) {
  if (timeZone && typeof Intl !== "undefined" && Intl.DateTimeFormat) {
    try {
      const parts = new Intl.DateTimeFormat("en-GB", {
        timeZone,
        hour: "2-digit",
        minute: "2-digit",
        hourCycle: "h23",
      }).formatToParts(dateObj);
      const hPart = parts.find((p) => p.type === "hour");
      const mPart = parts.find((p) => p.type === "minute");
      if (hPart && mPart) {
        return Number(hPart.value) + Number(mPart.value) / 60;
      }
    } catch (_err) {
      // fall back to local Date methods if timeZone string is invalid
    }
  }
  return dateObj.getHours() + dateObj.getMinutes() / 60;
}

function getDstAdjustedHorizonSteps(stepMin, days, firstStepDate, timeZone) {
  if (!Number.isFinite(stepMin) || stepMin <= 0 || !Number.isFinite(days) || days <= 0) {
    return null;
  }
  const baseDate = firstStepDate instanceof Date ? firstStepDate : new Date();
  const startMs = baseDate.getTime();
  const wholeDays = Math.max(1, Math.floor(days));
  const endDate = new Date(
    baseDate.getFullYear(),
    baseDate.getMonth(),
    baseDate.getDate() + wholeDays,
    baseDate.getHours(),
    baseDate.getMinutes(),
    0,
    0
  );
  // If browser timezone differs from configured IANA zone, compute calendar end via elapsed approx
  // refined to match local hour across DST transitions.
  let endMs = endDate.getTime();
  if (timeZone && typeof Intl !== "undefined" && Intl.DateTimeFormat) {
    try {
      const targetHour = getLocalHourInTimeZone(baseDate, timeZone);
      let guessMs = startMs + wholeDays * 24 * 3600 * 1000;
      for (const offsetMin of [0, -60, 60, -30, 30]) {
        const cand = new Date(guessMs + offsetMin * 60000);
        if (Math.abs(getLocalHourInTimeZone(cand, timeZone) - targetHour) < 1e-3) {
          endMs = cand.getTime();
          break;
        }
      }
    } catch (_err) {
      // keep endDate
    }
  }
  const totalMin = Math.max(stepMin, Math.round((endMs - startMs) / 60000));
  return Math.max(1, Math.round(totalMin / stepMin));
}

function getConfiguredHorizonSteps() {
  const stepEl = paramInputOf("optimization_time_step");
  const daysEl = paramInputOf("delta_forecast_daily");
  const stepMin = stepEl ? Number(stepEl.value) : 30;
  const days = daysEl ? Number(daysEl.value) : 1;
  if (!Number.isFinite(stepMin) || stepMin <= 0 || !Number.isFinite(days) || days <= 0) {
    return 48;
  }
  const tz = paramInputOf("time_zone")?.value || lastLoadedConfig?.time_zone || "";
  const steps = getDstAdjustedHorizonSteps(stepMin, days, new Date(), tz);
  return steps ?? Math.max(1, Math.round((days * 24 * 60) / stepMin));
}

function buildZonedStepDate(referenceDate, targetHour, timeZone) {
  const h = Math.floor(targetHour);
  const m = Math.round((targetHour % 1) * 60);
  if (timeZone && typeof Intl !== "undefined" && Intl.DateTimeFormat) {
    try {
      const fmt = new Intl.DateTimeFormat("en-GB", {
        timeZone,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hourCycle: "h23",
      });
      const getParts = (d) => {
        const map = {};
        for (const p of fmt.formatToParts(d)) {
          map[p.type] = Number(p.value);
        }
        return map;
      };
      const ref = getParts(referenceDate);
      const targetUtcMs = Date.UTC(ref.year, ref.month - 1, ref.day, h, m, 0, 0);
      let guessMs = targetUtcMs;
      for (let iter = 0; iter < 2; iter++) {
        const cur = getParts(new Date(guessMs));
        const curAsUtcMs = Date.UTC(cur.year, cur.month - 1, cur.day, cur.hour, cur.minute, 0, 0);
        const diffMs = targetUtcMs - curAsUtcMs;
        if (diffMs === 0) {
          break;
        }
        guessMs += diffMs;
      }
      return new Date(guessMs);
    } catch (_err) {
      // fall back to browser local timezone if timeZone string is invalid
    }
  }
  return new Date(
    referenceDate.getFullYear(),
    referenceDate.getMonth(),
    referenceDate.getDate(),
    h,
    m,
    0,
    0
  );
}

function buildDayNightTempArrays(opts) {
  const stepEl = paramInputOf("optimization_time_step");
  const daysEl = paramInputOf("delta_forecast_daily");
  const stepMin = stepEl ? Number(stepEl.value) : 30;
  const days = daysEl ? Number(daysEl.value) : 1;
  if (!Number.isFinite(stepMin) || stepMin <= 0) {
    return { error: "optimization_time_step must be a positive number of minutes." };
  }
  if (!Number.isFinite(days) || days <= 0) {
    return { error: "delta_forecast_daily must be a positive number of days." };
  }
  const rawStart = Number(opts.dayStartHour);
  const rawEnd = Number(opts.dayEndHour);
  const rawFirst = Number(opts.firstStepHour);
  const startH = Number.isFinite(rawStart) ? rawStart : 7;
  const endH = Number.isFinite(rawEnd) ? rawEnd : 22;
  const firstStepHour = Number.isFinite(rawFirst) ? rawFirst : 0;
  const tz =
    opts.timeZone || paramInputOf("time_zone")?.value || lastLoadedConfig?.time_zone || "";

  const now = opts.referenceDate instanceof Date ? opts.referenceDate : new Date();
  const firstStepDate = buildZonedStepDate(now, firstStepHour, tz);
  const horizon =
    Number.isFinite(Number(opts.horizon)) && Number(opts.horizon) > 0
      ? Number(opts.horizon)
      : getDstAdjustedHorizonSteps(stepMin, days, firstStepDate, tz);
  if (!Number.isFinite(horizon) || horizon < 1 || horizon > MAX_THERMAL_SCHEDULE_STEPS) {
    return {
      error: `Computed thermal schedule length (${horizon}) exceeds the supported limit (${MAX_THERMAL_SCHEDULE_STEPS} steps). Check optimization_time_step and delta_forecast_daily.`,
    };
  }
  const minDay = Number(opts.minDay);
  const minNight = Number(opts.minNight);
  const maxDay = Number(opts.maxDay);
  const maxNight = Number(opts.maxNight);

  const minArr = [];
  const maxArr = [];
  const startMs = firstStepDate.getTime();
  for (let i = 0; i < horizon; i++) {
    const stepDate = new Date(startMs + i * stepMin * 60000);
    const hour = getLocalHourInTimeZone(stepDate, tz);
    const isDay =
      startH <= endH ? hour >= startH && hour < endH : hour >= startH || hour < endH;
    minArr.push(isDay ? minDay : minNight);
    maxArr.push(isDay ? maxDay : maxNight);
  }
  return { minArr, maxArr };
}

function renderThermalLoadCards(builderEl, count) {
  while (thermalLoadStates.length < count) {
    thermalLoadStates.push({ kind: "standard", extra: {} });
  }

  let html = '<div class="thermal-builder-header">';
  html += "<h4>Thermal &amp; Heat Pump Model Configurator (<code>def_load_config</code>)</h4>";
  html +=
    '<p class="section-hint">Configure Simple Thermal Models (<code>thermal_config</code>) or Heat Pumps (<code>thermal_battery</code>) per load with a Day/Night comfort schedule generator anchored to your first optimization step hour.</p>';
  html += "</div>";

  if (count <= 0) {
    html += '<p class="muted-hint">Set Number of Loads &gt; 0 to configure deferrable or thermal loads.</p>';
    builderEl.innerHTML = html;
    return;
  }

  html += '<div class="thermal-cards-grid">';
  for (let idx = 0; idx < count; idx++) {
    const state = thermalLoadStates[idx] || { kind: "standard", extra: {} };
    const tc = state.thermal_config || {};
    const tb = state.thermal_battery || {};
    const extraKeys = state.extra ? Object.keys(state.extra) : [];
    const horizon = getConfiguredHorizonSteps();

    html += `<div class="thermal-load-card" data-load-index="${idx}">`;
    html += `<div class="thermal-load-card-head"><strong>Load ${idx}</strong>`;
    html += `<select class="thermal-kind-select form-select" data-load-index="${idx}" aria-label="Deferrable load ${idx} thermal model type">`;
    html += `<option value="standard"${state.kind === "standard" ? " selected" : ""}>Standard Deferrable Load ({})</option>`;
    html += `<option value="thermal_config"${state.kind === "thermal_config" ? " selected" : ""}>Simple Thermal Load (thermal_config)</option>`;
    html += `<option value="thermal_battery"${state.kind === "thermal_battery" ? " selected" : ""}>Heat Pump / Thermal Battery (thermal_battery)</option>`;
    if (state.kind === "passthrough") {
      html += `<option value="passthrough" selected>Custom / Preserved Entry (${escapeHtml(extraKeys.join(", "))})</option>`;
    }
    html += "</select></div>";

    if (state.kind === "passthrough") {
      html += `<p class="muted-hint">Preserving existing configuration keys: <code>${escapeHtml(
        extraKeys.join(", ")
      )}</code>. Switch model type above only if you intend to replace them.</p>`;
    } else if (state.kind === "standard") {
      html +=
        '<p class="muted-hint">Standard electrical load (no thermal envelope simulation attached; exports as <code>{}</code>).</p>';
    } else if (state.kind === "thermal_config") {
      const minStr = Array.isArray(tc.min_temperatures)
        ? tc.min_temperatures.map((v) => (v === null ? "null" : String(v))).join(", ")
        : "20";
      const maxStr = Array.isArray(tc.max_temperatures)
        ? tc.max_temperatures.map((v) => (v === null ? "null" : String(v))).join(", ")
        : "24";
      html += '<div class="thermal-fields-grid">';
      html += `<label>Heating rate (°C/h)<input type="number" step="any" data-field="heating_rate" value="${tc.heating_rate ?? 5.0}" aria-label="Load ${idx} heating rate"></label>`;
      html += `<label>Cooling constant<input type="number" step="any" data-field="cooling_constant" value="${tc.cooling_constant ?? 0.1}" aria-label="Load ${idx} cooling constant"></label>`;
      html += `<label>Thermal inertia (h)<input type="number" step="any" data-field="thermal_inertia" value="${tc.thermal_inertia ?? 0.0}" aria-label="Load ${idx} thermal inertia"></label>`;
      html += `<label>Start temp (°C)<input type="number" step="any" data-field="start_temperature" value="${tc.start_temperature ?? 20.0}" aria-label="Load ${idx} start temperature"></label>`;
      html += `<label>Sense<select data-field="sense" aria-label="Load ${idx} thermal sense"><option value="heat"${(tc.sense || "heat") === "heat" ? " selected" : ""}>heat</option><option value="cool"${tc.sense === "cool" ? " selected" : ""}>cool</option></select></label>`;
      html += "</div>";
      html += `<div class="thermal-schedule-helper">
        <span>Day/Night Comfort Schedule (${horizon} steps):</span>
        <div class="thermal-schedule-inputs">
          Start h: <input type="number" class="sched-first-hour" value="0" min="0" max="23" title="First optimization step hour (0-23)" aria-label="Load ${idx} first step hour">h |
          <input type="number" class="sched-day-start" value="7" min="0" max="23" title="Day start hour" aria-label="Load ${idx} day start hour">h–
          <input type="number" class="sched-day-end" value="22" min="0" max="23" title="Day end hour" aria-label="Load ${idx} day end hour">h |
          Min D/N: <input type="number" step="0.5" class="sched-min-day" value="20" aria-label="Load ${idx} day min temperature"> /
          <input type="number" step="0.5" class="sched-min-night" value="18" aria-label="Load ${idx} night min temperature"> °C |
          Max: <input type="number" step="0.5" class="sched-max-day" value="23" aria-label="Load ${idx} max temperature"> °C
          <button type="button" class="tool-btn apply-sched-btn" data-load-index="${idx}">Fill Schedule</button>
        </div>
      </div>`;
      html += `<div class="thermal-array-fields">
        <label>min_temperatures (comma-separated °C)<input type="text" data-field="min_temperatures" value="${escapeHtml(minStr)}" aria-label="Load ${idx} min temperatures"></label>
        <label>max_temperatures (comma-separated °C)<input type="text" data-field="max_temperatures" value="${escapeHtml(maxStr)}" aria-label="Load ${idx} max temperatures"></label>
      </div>`;
    } else if (state.kind === "thermal_battery") {
      const minStr = Array.isArray(tb.min_temperatures)
        ? tb.min_temperatures.map((v) => (v === null ? "null" : String(v))).join(", ")
        : "20";
      const maxStr = Array.isArray(tb.max_temperatures)
        ? tb.max_temperatures.map((v) => (v === null ? "null" : String(v))).join(", ")
        : "26";
      html += '<div class="thermal-fields-grid">';
      html += `<label>Supply temp (°C)<input type="number" step="any" data-field="supply_temperature" value="${tb.supply_temperature ?? 35.0}" aria-label="Load ${idx} supply temperature"></label>`;
      html += `<label>Buffer volume (m³/unit)<input type="number" step="any" data-field="volume" value="${tb.volume ?? 14.91}" aria-label="Load ${idx} volume"></label>`;
      html += `<label>Start temp (°C)<input type="number" step="any" data-field="start_temperature" value="${tb.start_temperature ?? 22.0}" aria-label="Load ${idx} start temperature"></label>`;
      html += `<label>Carnot efficiency<input type="number" step="any" data-field="carnot_efficiency" value="${tb.carnot_efficiency ?? 0.45}" aria-label="Load ${idx} Carnot efficiency"></label>`;
      html += `<label>U-value<input type="number" step="any" data-field="u_value" value="${tb.u_value ?? 0.326}" aria-label="Load ${idx} U-value"></label>`;
      html += `<label>Envelope area (m²)<input type="number" step="any" data-field="envelope_area" value="${tb.envelope_area ?? 649.4}" aria-label="Load ${idx} envelope area"></label>`;
      html += `<label>Ventilation rate<input type="number" step="any" data-field="ventilation_rate" value="${tb.ventilation_rate ?? 0.12}" aria-label="Load ${idx} ventilation rate"></label>`;
      html += `<label>Heated volume (m³)<input type="number" step="any" data-field="heated_volume" value="${tb.heated_volume ?? 573.5}" aria-label="Load ${idx} heated volume"></label>`;
      html += "</div>";
      html += `<div class="thermal-schedule-helper">
        <span>Day/Night Comfort Schedule (${horizon} steps):</span>
        <div class="thermal-schedule-inputs">
          Start h: <input type="number" class="sched-first-hour" value="0" min="0" max="23" title="First optimization step hour (0-23)" aria-label="Load ${idx} first step hour">h |
          <input type="number" class="sched-day-start" value="7" min="0" max="23" title="Day start hour" aria-label="Load ${idx} day start hour">h–
          <input type="number" class="sched-day-end" value="22" min="0" max="23" title="Day end hour" aria-label="Load ${idx} day end hour">h |
          Min D/N: <input type="number" step="0.5" class="sched-min-day" value="20" aria-label="Load ${idx} day min temperature"> /
          <input type="number" step="0.5" class="sched-min-night" value="18.5" aria-label="Load ${idx} night min temperature"> °C |
          Max: <input type="number" step="0.5" class="sched-max-day" value="26" aria-label="Load ${idx} max temperature"> °C
          <button type="button" class="tool-btn apply-sched-btn" data-load-index="${idx}">Fill Schedule</button>
        </div>
      </div>`;
      html += `<div class="thermal-array-fields">
        <label>min_temperatures (comma-separated °C)<input type="text" data-field="min_temperatures" value="${escapeHtml(minStr)}" aria-label="Load ${idx} min temperatures"></label>
        <label>max_temperatures (comma-separated °C)<input type="text" data-field="max_temperatures" value="${escapeHtml(maxStr)}" aria-label="Load ${idx} max temperatures"></label>
      </div>`;
    }
    html += "</div>";
  }
  html += "</div>";
  builderEl.innerHTML = html;

  builderEl.querySelectorAll(".thermal-kind-select").forEach((sel) => {
    sel.addEventListener("change", () => {
      const i = Number(sel.dataset?.loadIndex);
      const kind = sel.value;
      const existingExtra = thermalLoadStates[i] ? thermalLoadStates[i].extra || {} : {};
      if (kind === "passthrough") {
        thermalLoadStates[i] = { kind: "passthrough", extra: existingExtra };
      } else if (kind === "standard") {
        thermalLoadStates[i] = { kind: "standard", extra: {} };
      } else if (kind === "thermal_config") {
        thermalLoadStates[i] = {
          kind: "thermal_config",
          extra: existingExtra,
          thermal_config: thermalLoadStates[i].thermal_config || {
            heating_rate: 5.0,
            cooling_constant: 0.1,
            thermal_inertia: 0.0,
            start_temperature: 20.0,
            sense: "heat",
            min_temperatures: [20.0],
            max_temperatures: [24.0],
          },
        };
      } else {
        thermalLoadStates[i] = {
          kind: "thermal_battery",
          extra: existingExtra,
          thermal_battery: thermalLoadStates[i].thermal_battery || {
            supply_temperature: 35.0,
            volume: 14.91,
            start_temperature: 22.0,
            carnot_efficiency: 0.45,
            u_value: 0.326,
            envelope_area: 649.4,
            ventilation_rate: 0.12,
            heated_volume: 573.5,
            min_temperatures: [20.0],
            max_temperatures: [26.0],
          },
        };
      }
      renderThermalLoadCards(builderEl, count);
      updateConfigSystemOverviewBadges(lastLoadedConfig);
    });
  });

  builderEl.querySelectorAll(".thermal-load-card").forEach((card) => {
    const idx = Number(card.dataset?.loadIndex);
    card.querySelectorAll("[data-field]").forEach((input) => {
      const markDirtyAndSync = () => {
        if (input.dataset) {
          input.dataset.dirty = "1";
        }
        syncSingleThermalState(card, idx);
      };
      input.addEventListener("change", markDirtyAndSync);
      input.addEventListener("input", markDirtyAndSync);
    });
    const schedBtn = card.querySelector(".apply-sched-btn");
    if (schedBtn) {
      schedBtn.addEventListener("click", () => {
        const minDay = Number(card.querySelector(".sched-min-day").value);
        const minNight = Number(card.querySelector(".sched-min-night").value);
        const maxVal = Number(card.querySelector(".sched-max-day").value);
        const built = buildDayNightTempArrays({
          firstStepHour: Number(card.querySelector(".sched-first-hour")?.value ?? 0),
          dayStartHour: Number(card.querySelector(".sched-day-start").value),
          dayEndHour: Number(card.querySelector(".sched-day-end").value),
          minDay,
          minNight,
          maxDay: maxVal,
          maxNight: maxVal,
        });
        if (built.error) {
          errorAlert(built.error);
          return;
        }
        const minInput = card.querySelector('[data-field="min_temperatures"]');
        const maxInput = card.querySelector('[data-field="max_temperatures"]');
        if (minInput) {
          minInput.value = built.minArr.join(", ");
          if (minInput.dataset) {
            minInput.dataset.dirty = "1";
          }
        }
        if (maxInput) {
          maxInput.value = built.maxArr.join(", ");
          if (maxInput.dataset) {
            maxInput.dataset.dirty = "1";
          }
        }
        syncSingleThermalState(card, idx);
      });
    }
  });
}

function syncSingleThermalState(card, idx) {
  const state = thermalLoadStates[idx];
  if (!state || state.kind === "standard" || state.kind === "passthrough") {
    return;
  }
  const bucket = state[state.kind] || {};
  card.querySelectorAll("[data-field]").forEach((el) => {
    const field = el.dataset?.field;
    if (field === "min_temperatures" || field === "max_temperatures") {
      // Preserve loaded temperature arrays verbatim unless this field was explicitly edited
      if (Array.isArray(bucket[field]) && el.dataset?.dirty !== "1") {
        return;
      }
      bucket[field] = parseTempArrayText(el.value);
    } else if (field === "sense") {
      bucket[field] = el.value;
    } else {
      const n = Number.parseFloat(el.value);
      if (Number.isFinite(n)) {
        bucket[field] = n;
      }
    }
  });
  state[state.kind] = bucket;
}

function parseThermalLoadStateItem(item) {
  const isObj = Boolean(item && typeof item === "object" && !Array.isArray(item));
  const extra = {};
  if (isObj) {
    for (const k of Object.keys(item)) {
      if (k !== "thermal_config" && k !== "thermal_battery") {
        extra[k] = item[k];
      }
    }
    if (item.thermal_battery) {
      return {
        kind: "thermal_battery",
        extra,
        thermal_battery: { ...item.thermal_battery },
      };
    }
    if (item.thermal_config) {
      return {
        kind: "thermal_config",
        extra,
        thermal_config: { ...item.thermal_config },
      };
    }
  }
  if (Object.keys(extra).length > 0) {
    return { kind: "passthrough", extra };
  }
  return { kind: "standard", extra: {} };
}

function setupThermalLoadsBuilder(config) {
  const builderEl = document.getElementById("thermal-loads-builder");
  const countInput = document.getElementById("number_of_deferrable_loads");
  if (!builderEl || !countInput) {
    return;
  }
  const rawList = Array.isArray(config?.def_load_config) ? config.def_load_config : [];
  const count = Math.max(0, Number.parseInt(countInput.value, 10) || 0);
  thermalLoadStates = [];
  for (let i = 0; i < Math.max(count, rawList.length); i++) {
    thermalLoadStates.push(parseThermalLoadStateItem(rawList[i]));
  }
  renderThermalLoadCards(builderEl, count);

  if (
    typeof countInput.addEventListener === "function" &&
    !countInput._boundThermalBuilder &&
    !countInput.dataset?.boundThermalBuilder
  ) {
    countInput._boundThermalBuilder = true;
    if (countInput.dataset) {
      countInput.dataset.boundThermalBuilder = "1";
    }
    countInput.addEventListener("input", () => {
      const nextCount = Math.max(0, Number.parseInt(countInput.value, 10) || 0);
      renderThermalLoadCards(builderEl, nextCount);
    });
  }
}

function buildThermalLoadsConfig() {
  const builderEl = document.getElementById("thermal-loads-builder");
  const countInput = document.getElementById("number_of_deferrable_loads");
  if (!builderEl || !countInput) {
    return null;
  }
  const count = Math.max(0, Number.parseInt(countInput.value, 10) || 0);
  const cards = builderEl.querySelectorAll(".thermal-load-card");
  cards.forEach((card) => {
    const idx = Number(card.dataset?.loadIndex);
    syncSingleThermalState(card, idx);
  });
  const result = [];
  for (let i = 0; i < count; i++) {
    const state = thermalLoadStates[i];
    if (state?.kind === "thermal_config") {
      result.push({
        ...state.extra,
        thermal_config: { ...state.thermal_config },
      });
    } else if (state?.kind === "thermal_battery") {
      result.push({
        ...state.extra,
        thermal_battery: { ...state.thermal_battery },
      });
    } else if (state?.kind === "passthrough") {
      result.push({ ...state.extra });
    } else {
      result.push({ ...state?.extra });
    }
  }
  return result;
}

function updateConfigSystemOverviewBadges(fallbackConfig) {
  const strip = document.getElementById("system-overview-badges");
  if (!strip) {
    return;
  }
  const readBool = (id, defVal) => {
    const el = paramInputOf(id) || document.getElementById(id);
    if (el && el.type === "checkbox") return Boolean(el.checked);
    return fallbackConfig && id in fallbackConfig ? Boolean(fallbackConfig[id]) : defVal;
  };
  const readNum = (id, defVal) => {
    const el = paramInputOf(id) || document.getElementById(id);
    if (el && el.value !== "") {
      const v = Number(el.value);
      if (Number.isFinite(v)) return v;
    }
    return fallbackConfig && id in fallbackConfig ? Number(fallbackConfig[id]) : defVal;
  };
  const readStr = (id, defVal) => {
    const el = paramInputOf(id) || document.getElementById(id);
    if (el?.value) return String(el.value);
    return fallbackConfig?.[id] ? String(fallbackConfig[id]) : defVal;
  };

  const badges = [];
  if (readBool("set_use_pv", false)) {
    badges.push({ cls: "sys-pv", text: "PV Plant (" + readStr("weather_forecast_method", "csv") + ")" });
  } else {
    badges.push({ cls: "sys-muted", text: "No PV" });
  }
  if (readBool("set_use_battery", false)) {
    const nb = Math.max(1, readNum("number_of_batteries", 1));
    badges.push({ cls: "sys-batt", text: nb > 1 ? nb + " Batteries" : "1 Battery" });
  } else {
    badges.push({ cls: "sys-muted", text: "No Battery" });
  }
  const ndef = Math.max(0, readNum("number_of_deferrable_loads", 0));
  badges.push({
    cls: ndef > 0 ? "sys-loads" : "sys-muted",
    text: ndef + " Deferrable Load" + (ndef === 1 ? "" : "s"),
  });
  let thermalCount = 0;
  const isBoxMode = document.getElementById("config-box") !== null;
  if (isBoxMode) {
    const rawDefCfg =
      fallbackConfig && Array.isArray(fallbackConfig.def_load_config)
        ? fallbackConfig.def_load_config
        : [];
    for (let i = 0; i < Math.min(ndef, rawDefCfg.length); i++) {
      const item = rawDefCfg[i];
      if (
        item &&
        typeof item === "object" &&
        (item.thermal_config || item.thermal_battery || item.thermal_source)
      ) {
        thermalCount += 1;
      }
    }
  } else {
    for (let i = 0; i < Math.min(ndef, thermalLoadStates.length); i++) {
      if (thermalLoadStates[i] && thermalLoadStates[i].kind !== "standard") {
        thermalCount += 1;
      }
    }
  }
  if (thermalCount > 0) {
    badges.push({
      cls: "sys-thermal",
      text: thermalCount + " Thermal Load" + (thermalCount === 1 ? "" : "s"),
    });
  }
  const nInv = Math.max(1, readNum("number_of_inverters", 1));
  badges.push({
    cls: "sys-neutral",
    text: nInv > 1 ? nInv + " Inverters" : readBool("inverter_is_hybrid", false) ? "Hybrid Inverter" : "1 Inverter",
  });
  const capContainer = document.getElementById("capacity_cost_per_kw");
  let hasCapTariff = false;
  if (capContainer && typeof capContainer.getElementsByClassName === "function") {
    const capInputs = Array.from(capContainer.getElementsByClassName("param_input"));
    if (capInputs.length > 0) {
      hasCapTariff = capInputs.some((inp) => Number(inp.value) > 0);
    }
  } else if (fallbackConfig && "capacity_cost_per_kw" in fallbackConfig) {
    const rawCap = fallbackConfig.capacity_cost_per_kw;
    hasCapTariff = Array.isArray(rawCap)
      ? rawCap.some((v) => Number(v) > 0)
      : Number(rawCap) > 0;
  }
  if (hasCapTariff) {
    badges.push({ cls: "sys-tariff", text: "Capacity Tariff" });
  }
  badges.push(
    { cls: "sys-neutral", text: "Objective: " + readStr("costfun", "profit") },
    { cls: "sys-load", text: "Load: " + readStr("load_forecast_method", "naive") },
    { cls: "sys-neutral", text: readNum("optimization_time_step", 30) + " min step" }
  );
  if (readStr("optimization_backend", "cvxpy") === "dantzig_wolfe") {
    badges.push({ cls: "sys-fed", text: "Federated Coordinator" });
  }
  strip.innerHTML =
    '<span class="system-overview-title">Configured system:</span>' +
    badges
      .map((b) => `<span class="sys-badge ${b.cls}">${escapeHtml(b.text)}</span>`)
      .join("");
}
