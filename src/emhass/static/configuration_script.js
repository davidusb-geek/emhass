//javascript file for processing configuration page

//Files
//param_definitions.json : stores information about parameters (E.g. their defaults, their type, and what section to be in)
//configuration_list.html : template html to act as a base for the list view. (Params get dynamically added after)

//on page reload
window.onload = async function () {
  ///fetch configuration parameters from json file
  param_definitions = await getParamDefinitions();
  //obtain configuration from emhass
  config = await obtainConfig();
  //obtain list template html to render parameters in list view (as input items)
  list_html = await getListHTML();
  //load list parameter page (default)
  loadConfigurationListPage(param_definitions, config, list_html);

  //add event listener to save button
  document
    .getElementById("save")
    .addEventListener("click", () => saveConfiguration(param_definitions));

  //add defaults listener to save button
  document
    .getElementById("defaults")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, true)
    );

  //add json listener to save button (toggle between json box and list view)
  document
    .getElementById("json-toggle")
    .addEventListener("click", () =>
      ToggleView(param_definitions, list_html, false)
    );
};

//obtain file containing information about parameters
async function getParamDefinitions() {
  const response = await fetch(`static/data/param_definitions.json`);
  if (response.status !== 200 && response.status !== 201) {
    //alert error in alert box
    errorAlert("Unable to obtain definitions file");
  }
  const param_definitions = await response.json();
  return await param_definitions;
}

//obtain emhass built config
async function obtainConfig() {
  config = {};
  response = await fetch(`/get-config`, {
    method: "GET",
  });
  blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  if (response.status !== 200 && response.status !== 201) {
    errorAlert("Unable to obtain config file");
  }
  return config;
}

//obtain emhass default config
async function ObtainDefaultConfig() {
  config = {};
  response = await fetch(`/get-config/defaults`, {
    method: "GET",
  });
  blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  if (response.status !== 200 && response.status !== 201) {
    errorAlert("Unable to obtain default config file");
  }

  return config;
}

//get html data from configuration_list.html (list template)
async function getListHTML() {
  const response = await fetch(`static/configuration_list.html`);
  blob = await response.blob(); //get data blob
  htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  showChangeStatus(response.status, [
    "Unable to obtain configuration_list html file",
  ]);
  return await htmlTemplateData;
}

//load list configuration page/form
function loadConfigurationListPage(param_definitions, config, list_html) {
  //list parameters used in the section headers
  header_input_list = ["set_use_battery", "number_of_deferrable_loads"];

  //get the main container and append list template as a base
  document.getElementById("configurationContainer").innerHTML = list_html;

  //loop though configuration sections ('Local','System','Tariff','Solar System (PV)') in json file, build sections
  for (var section in param_definitions) {
    //build one section at a time
    buildParamContainers(
      section,
      param_definitions[section],
      config,
      header_input_list
    );

    //after sections have been built
    //add event listeners for section header inputs
    for (header_input_param of header_input_list) {
      if (param_definitions[section].hasOwnProperty(header_input_param)) {
        //grab default from definitions file
        value = param_definitions[section][header_input_param]["default_value"];
        //find input element (parameter name as the input element ID)
        header_input_element = document.getElementById(header_input_param);
        //add listener to element to input
        header_input_element.addEventListener("input", (e) =>
          headerElement(e.target, param_definitions, config)
        );
        //check EMHASS config contains a stored param value
        value = checkConfigParam(value, config, header_input_param);
        //set value of input
        header_input_element.value = value;
        //checkboxes (for Booleans) use checked instead of value
        if (header_input_element.tagName == "checkbox") {
          header_input_element.checked = value;
        }
        //manually trigger header element event listener for initial state
        headerElement(header_input_element, param_definitions, config);
      }
    }
  }
}

//build sections body, param containers (containing param input)
function buildParamContainers(
  section,
  section_parameters_definitions,
  config,
  header_input_list
) {
  //get the section container
  SectionContainer = document.getElementById(section);
  //get the body container inside the section
  SectionParamElement = SectionContainer.getElementsByClassName("section-body");

  //loop though parameters in definition file, generate and append param containers for the section
  for (const [
    parameter_definition_name,
    parameter_definition_object,
  ] of Object.entries(section_parameters_definitions)) {
    //if type array.* and not in "Deferrable Loads" section, add plus and minus buttons
    array_buttons = "";
    if (
      parameter_definition_object["input"].search("array.") > -1 &&
      section != "Deferrable Loads"
    ) {
      array_buttons = `
                  <button type="button" class="input-plus ${parameter_definition_name}">+</button>
                  <button type="button" class="input-minus ${parameter_definition_name}">-</button>
                  <br>
                  `;
    }
    //check if param is set in the section header, if so skip param container html append
    if (header_input_list.includes(parameter_definition_name)) {
      continue;
    }

    //generates and appends param container
    SectionParamElement[0].innerHTML += `
          <div class="param" id="${parameter_definition_name}">
             <h5>${
               parameter_definition_object["friendly_name"]
             }:</h5> <i>${parameter_definition_name}</i> </br>
              ${array_buttons}
             <div class="param-input"> 
                  ${buildParamElement(
                    parameter_definition_object,
                    parameter_definition_name,
                    config
                  )}
             </div>
              <p>${parameter_definition_object["Description"]}</p>
          </div>
          `;
  }

  //add button (array plus) event listeners
  let plus = SectionContainer.querySelectorAll(".input-plus");
  plus.forEach(function (answer) {
    answer.addEventListener("click", () =>
      plusElements(answer.classList[1], param_definitions, section, {})
    );
  });

  //subtract button (array minus) event listeners
  let minus = SectionContainer.querySelectorAll(".input-minus");
  minus.forEach(function (answer) {
    answer.addEventListener("click", () => minusElements(answer.classList[1]));
  });

  //check boxes that should be ticked (check value of input and match to checked)
  let checkbox = document.querySelectorAll("input[type='checkbox']");
  checkbox.forEach(function (answer) {
    let value = answer.value === "true";
    answer.checked = value;
  });

  //loop though sections params again, check if param has a requirement, if so add a listener to the required param input
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
      // get param that has requirement
      const param_element = document.getElementById(parameter_definition_name);
      //obtain param inputs, on change, trigger function
      requirement_inputs =
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
          //manually run function to gain initial state
          checkRequirements(input, param_element, requirement_value);
        }
      }
    }
  }
}

//create html input element/s for a param container (called by buildParamContainers)
function buildParamElement(
  parameter_definition_object,
  parameter_definition_name,
  config
) {
  var type = "";
  var inputs = "";
  var type_specific_html = "";
  var type_specific_html_end = "";

  //switch statement to adjust generated html according to its data type
  switch (parameter_definition_object["input"]) {
    case "array.int":
    case "int":
      type = "number";
      placeholder = parseInt(parameter_definition_object["default_value"]);
      break;
    case "array.string":
    case "string":
      type = "text";
      placeholder = parameter_definition_object["default_value"];
      break;
    case "array.time":
    case "time":
      type = "time";
      break;
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
      placeholder = parameter_definition_object["default_value"] === "true";
      break;
    case "array.float":
    case "float":
      type = "number";
      placeholder = parseFloat(parameter_definition_object["default_value"]);
      break;
    case "select":
      break;
  }

  //check default values saved in param definitions
  value = parameter_definition_object["default_value"];
  //check if a param value is saved in the config file (if so overwrite definition default)
  value = checkConfigParam(value, config, parameter_definition_name);

  //generate and return param input html,
  //check if param value is an object if so treat value as an array of values
  if (typeof value !== "object") {
    //if select, generate and return select instead of input
    if (parameter_definition_object["input"] == "select") {
      let inputs = `<select class="param_input">`;
      for (const options of parameter_definition_object["select_options"]) {
        inputs += `<option value="${options}">${options}</option>`;
      }
      inputs += `</select>`;
      return inputs;
    }
    // generate param input html and return
    else {
      return `
          ${type_specific_html}
          <input class="param_input" type="${type}" value=${value} placeholder=${parameter_definition_object["default_value"]}>
          ${type_specific_html_end}
          `;
    }
  }
  // else (value isn't a object) loop though values, generate inputs and and return
  else {
    //for items such as load_peak_hour_periods (object of array objects)
    if (typeof Object.values(value)[0] === "object") {
      for (param of Object.values(value)) {
        for (items of Object.values(param)) {
          inputs += `<input class="param_input" type="${type}" value=${
            Object.values(items)[0]
          } placeholder=${Object.values(items)[0]}>`;
        }
        inputs += `</br>`;
      }
      return inputs;
    }
    // array of values/objects
    else {
      let inputs = "";
      for (param of value) {
        inputs += `
          ${type_specific_html}
          <input class="param_input" type="${type}" value=${param} placeholder=${parameter_definition_object["default_value"]}>
          ${type_specific_html_end}
          `;
      }
      return inputs;
    }
  }
}

//add param elements (for type array)
function plusElements(
  parameter_definition_name,
  param_definitions,
  section,
  config
) {
  param_element = document.getElementById(parameter_definition_name);
  param_input_container =
    param_element.getElementsByClassName("param-input")[0];
  // Add a copy of the param element
  param_input_container.innerHTML += buildParamElement(
    param_definitions[section][parameter_definition_name],
    parameter_definition_name,
    config
  );
}

//Remove param elements (minimum 1)
function minusElements(param) {
  param_inputs = document.getElementById(param).getElementsByTagName("input");

  //verify if input is a boolean (if so remove switch with input)
  if (param_inputs[param_inputs.length - 1].parentNode.tagName === "LABEL") {
    param_input = param_inputs[param_inputs.length - 1].parentNode;
  } else {
    param_input = param_inputs[param_inputs.length - 1];
  }

  if (param_inputs.length > 1) {
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
  } else {
    if (param_element.classList.contains("requirement-disable")) {
      param_element.classList.remove("requirement-disable");
    }
  }
}

//retrieve header inputs and execute accordingly
function headerElement(element, param_definitions, config) {
  switch (element.id) {
    //if set_use_battery, add or remove battery section (inc. params)
    case "set_use_battery":
      param_container = element
        .closest(".section-card")
        .getElementsByClassName("section-body")[0];
      param_container.innerHTML = "";
      if (element.checked) {
        buildParamContainers("Battery", param_definitions["Battery"], config, [
          "set_use_battery",
        ]);
        element.checked = true;
      }
      break;
    //if number_of_deferrable_loads, number of inputs per param in Deferrable Loads section should add up to number_of_deferrable_loads in header
    case "number_of_deferrable_loads":
      param_container = element
        .closest(".section-card")
        .getElementsByClassName("section-body")[0];
      param_list = param_container.getElementsByClassName("param");
      difference =
        parseInt(element.value) -
        param_container.firstElementChild.querySelectorAll("input").length;
      //add elements
      if (difference > 0) {
        for (let i = difference; i >= 1; i--) {
          for (const param of param_list) {
            //append element, do not pass config to obtain default parameter from definitions file
            plusElements(param.id, param_definitions, "Deferrable Loads", {});
          }
        }
      }
      //subtract elements
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

//checks parameter in config, updates value if exists
function checkConfigParam(value, config, parameter_definition_name) {
  isArray = false;
  if (config !== null && config !== undefined) {
    //check if parameter has a saved value
    if (parameter_definition_name in config) {
      value = config[parameter_definition_name];
    }
    //check values saved in config are dict arrays (E.g. sensor_replace_zero in list_sensor_replace_zero)
    //extract values and return object array
    if ("list_" + parameter_definition_name in config) {
      isArray = true;
      // extract parameter values from object array
      value = config["list_" + parameter_definition_name].map(function (a) {
        return a[parameter_definition_name];
      });
    }
  }
  return value;
}

//send parameter input values to EMHASS, to save to config.json and param.pkl
async function saveConfiguration(param_definitions) {
  //start wth none
  config = {};
  //check if page is in list or box view
  config_box_element = document.getElementById("config-box");

  //if true, in list view
  if (config_box_element === null) {
    //retrieve params by looping though param_definitions list (loop through configuration sections)
    for (var [section_name, section_object] of Object.entries(
      param_definitions
    )) {
      //loop through parameters
      for (var [
        parameter_definition_name,
        parameter_definition_object,
      ] of Object.entries(section_object)) {
        let param_values = [];
        let param_array = false;
        // get param container
        param_element = document.getElementById(parameter_definition_name);
        //extract inputs from param
        if (param_element != null) {
          //check if param_element is also param_input (ex. for header param)
          if (param_element.tagName == "INPUT" && param_inputs.length === 0) {
            param_inputs = [param_element];
          } else {
            param_inputs = param_element.getElementsByClassName("param_input");
          }

          //obtain param input type from param_definitions, check if param should be formatted as an array
          param_array = Boolean(
            !parameter_definition_object["input"].search("array")
          );
          // loop though param_inputs, extract the element/s values
          for (var input of param_inputs) {
            switch (input.type) {
              case "number":
                param_values.push(parseFloat(input.value));
                break;
              case "checkbox":
                param_values.push(input.checked);
                break;
              default:
                param_values.push(input.value);
                break;
            }
          }
          //build parameters using values from inputs
          if (param_array) {
            //load_peak_hour_periods (object of objects )
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
            }
            //array list (array of objects)
            else {
              config["list_" + parameter_definition_name] = [];
              for (const value of param_values) {
                config["list_" + parameter_definition_name].push({
                  [parameter_definition_name]: value,
                });
              }
            }
          }
          //single value
          if (!param_array && param_values.length) {
            config[parameter_definition_name] = param_values[0];
          }
        }
      }
    }
  }
  //in box view
  else {
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

  //send built config to emhass
  const response = await fetch(`/set-config`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(config),
  });
  showChangeStatus(response.status, await response.json());
}

//Toggle between box (json) and list view
async function ToggleView(param_definitions, list_html, default_reset) {
  let selected = "";
  config = {};

  //find out if list or box view is active
  configuration_container = document.getElementById("configurationContainer");
  // if section-cards (config sections/list) exists
  config_card = configuration_container.getElementsByClassName("section-card");
  //selected view (0 = box)
  selected_view = Boolean(config_card.length);

  //if default_reset is passed do not switch views, instead reinitialize view with default params
  if (default_reset) {
    selected_view = !selected_view;
    //obtain default config as config (when pressing the default button)
    config = await ObtainDefaultConfig();
  } else {
    //obtain latest config
    config = await obtainConfig();
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
  switch (selected) {
    case "box":
      //load list
      loadConfigurationListPage(param_definitions, config, list_html);
      break;
    case "list":
      //load box
      loadConfigurationBoxPage(config);
      break;
  }
}

//load box (json textarea) html
async function loadConfigurationBoxPage(config) {
  configuration_container.innerHTML = `
      <textarea id="config-box" rows="30" placeholder="{}"></textarea>
      `;
  //set created textarea box with retrieved config
  document.getElementById("config-box").innerHTML = JSON.stringify(
    config,
    null,
    2
  );
}

//function in control of status icons and alert box from a fetch request
async function showChangeStatus(status, logJson) {
  var loading = document.getElementById("loader"); //element showing statuses
  if (status === "remove") {
    //remove all
    loading.innerHTML = "";
    loading.classList.remove("loading");
  } else if (status === "loading") {
    //show loading logo
    loading.innerHTML = "";
    loading.classList.add("loading"); //append class with loading animation styling
  } else if (status === 201) {
    //if status is 201, then show a tick
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=tick>&#x2713;</p>`;
  } else if (status === 200) {
    //if status is 201, then show a tick
    loading.innerHTML = "";
    loading.classList.remove("loading");
  } else {
    //then show a cross
    loading.classList.remove("loading");
    loading.innerHTML = `<p class=cross>&#x292C;</p>`; //show cross icon to indicate an error
    if (logJson.length != 0) {
      document.getElementById("alert-text").textContent =
        "\r\n\u2022 " + logJson.join("\r\n\u2022 "); //show received log data in alert box
      document.getElementById("alert").style.display = "block";
      document.getElementById("alert").style.textAlign = "left";
    }
  }
}

//simple function to write text to the alert box
async function errorAlert(text) {
  document.getElementById("alert-text").textContent = "\r\n" + text + "\r\n";
  document.getElementById("alert").style.display = "block";
  document.getElementById("alert").style.textAlign = "left";
  return 0;
}

//check config works
//remove logging
//loading
//tick and cross sizing
//add config to .gitignore
//remove old objects
//clean up css
//array values (refine pv peak)
//error in send or retrive (alert box)
//config path in data folder
//cli paths
//add redudency?
