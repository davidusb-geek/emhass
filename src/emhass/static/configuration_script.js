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

//on page reload
window.onload = async function () {
  ///fetch configuration parameters from definitions json file
  param_definitions = await getParamDefinitions();
  //obtain configuration from emhass (pull)
  config = await obtainConfig();
  //obtain configuration_list.html html as a template to dynamically to render parameters in a list view (parameters as input items)
  list_html = await getListHTML();
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
};

//obtain file containing information about parameters (definitions)
async function getParamDefinitions() {
  const response = await fetch(`static/data/param_definitions.json`);
  if (response.status !== 200 && response.status !== 201) {
    //alert error in alert box
    errorAlert("Unable to obtain definitions file");
    return {};
  }
  const param_definitions = await response.json();
  return await param_definitions;
}

//obtain emhass config (from saved params extracted/simplified into the config format)
async function obtainConfig() {
  config = {};
  const response = await fetch(`get-config`, {
    method: "GET",
  });
  response_status = await response.status; //return status
  //if request failed
  if (response_status !== 200 && response_status !== 201) {
    showChangeStatus(response_status, await response.json());
    return {};
  }
  //else extract json rom data
  blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  showChangeStatus(response_status, {});
  return config;
}

//obtain emhass default config (to present the default parameters in view)
async function ObtainDefaultConfig() {
  config = {};
  const response = await fetch(`get-config/defaults`, {
    method: "GET",
  });
  //if request failed
  response_status = await response.status; //return status
  if (response_status !== 200 && response_status !== 201) {
    showChangeStatus(response_status, await response.json());
    return {};
  }
  //else extract json rom data
  blob = await response.blob(); //get data blob
  config = await new Response(blob).json(); //obtain json from blob
  showChangeStatus(response_status, {});
  return config;
}

//get html data from configuration_list.html (list template)
async function getListHTML() {
  const response = await fetch(`static/configuration_list.html`);
  if (response.status !== 200 && response.status !== 201) {
    errorAlert("Unable to obtain configuration_list.html file");
    return {};
  }
  blob = await response.blob(); //get data blob
  htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  return await htmlTemplateData;
}

//load list configuration view
function loadConfigurationListView(param_definitions, config, list_html) {
  if (list_html == null || config == null || param_definitions == null) {
    return 1;
  }

  //list parameters used in the section headers
  header_input_list = ["set_use_battery", "number_of_deferrable_loads"];

  //get the main container and append list template html
  document.getElementById("configuration-container").innerHTML = list_html;

  //loop though configuration sections ('Local','System','Tariff','Solar System (PV)') in definitions file
  for (var section in param_definitions) {
    // build each section by adding parameters with their corresponding input elements
    buildParamContainers(
      section,
      param_definitions[section],
      config,
      header_input_list
    );

    //after sections have been built, add event listeners for section header inputs
    //loop though headers
    for (header_input_param of header_input_list) {
      if (param_definitions[section].hasOwnProperty(header_input_param)) {
        //grab default from definitions file
        value = param_definitions[section][header_input_param]["default_value"];
        //find input element (using the parameter name as the input element ID)
        header_input_element = document.getElementById(header_input_param);
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
}

//build sections body, containing parameter/param containers (containing parameter/param inputs)
function buildParamContainers(
  section,
  section_parameters_definitions,
  config,
  header_input_list
) {
  //get the section container element
  SectionContainer = document.getElementById(section);
  //get the body container inside the section (where the parameters will be appended)
  SectionParamElement = SectionContainer.getElementsByClassName("section-body");
  if (SectionContainer == null || SectionParamElement.length == 0) {
    console.error("Unable to find Section container or Section Body");
    return 0;
  }

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

    //if parameter type == array.* and not in "Deferrable Loads" section, append plus and minus buttons in param div
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

    //generates and appends param container into section
    //buildParamElement() builds the parameter input/s and returns html to append in param-input
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

  //After looping though, build and appending the parameters in the corresponding section:
  //create add button (array plus) event listeners
  let plus = SectionContainer.querySelectorAll(".input-plus");
  plus.forEach(function (answer) {
    answer.addEventListener("click", () =>
      plusElements(answer.classList[1], param_definitions, section, {})
    );
  });

  //create subtract button (array minus) event listeners
  let minus = SectionContainer.querySelectorAll(".input-minus");
  minus.forEach(function (answer) {
    answer.addEventListener("click", () => minusElements(answer.classList[1]));
  });

  //check initial checkbox state, check "value" of input and match to "checked" value
  let checkbox = document.querySelectorAll("input[type='checkbox']");
  checkbox.forEach(function (answer) {
    let value = answer.value === "true";
    answer.checked = value;
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
          //manually run function to gain initial param element initial state
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

  //switch statement to adjust generated html according to the parameter data type (definitions in definitions file)
  switch (parameter_definition_object["input"]) {
    case "array.int":
    //number
    case "int":
      type = "number";
      placeholder = parseInt(parameter_definition_object["default_value"]);
      break;
    case "array.float":
    case "float":
      type = "number";
      placeholder = parseFloat(parameter_definition_object["default_value"]);
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
      placeholder = parameter_definition_object["default_value"] === "true";
      break;
    //selects (pick)
    case "select":
      //format selects later
      break;
  }

  //check default values saved in param definitions
  //definitions default value is used if none is found in the configs, or an array element has been added in the ui (deferrable load number increase or plus button pressed)
  value = parameter_definition_object["default_value"];
  //check if a param value is saved in the config file (if so overwrite definition default)
  value = checkConfigParam(value, config, parameter_definition_name);

  //generate and return param input html,
  //check if param value is not an object, if so assume its a single value.
  if (typeof value !== "object") {
    //if select, generate and return select elements instead of input
    if (parameter_definition_object["input"] == "select") {
      let inputs = `<select class="param_input">`;
      for (const options of parameter_definition_object["select_options"]) {
        selected = ""
        //if item in select is the same as the config value, then append "selected" tag
        if (options==value) {selected = `selected="selected"`}
        inputs += `<option ${selected}>${options}</option>`;
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
  // else if object, loop though array of values, generate input element per value, and and return
  else {
    //for items such as load_peak_hour_periods (object of objects with arrays)
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
    // array of values
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

//add param inputs in param div container (for type array)
function plusElements(
  parameter_definition_name,
  param_definitions,
  section,
  config
) {
  param_element = document.getElementById(parameter_definition_name);
  if (param_element == null) {
    console.log(
      "Unable to find " + parameter_definition_name + " param div container"
    );
    return 1;
  }
  param_input_container =
    param_element.getElementsByClassName("param-input")[0];
  // Add a copy of the param element
  param_input_container.innerHTML += buildParamElement(
    param_definitions[section][parameter_definition_name],
    parameter_definition_name,
    config
  );
}

//Remove param inputs in param div container (minimum 1)
function minusElements(param) {
  param_element = document.getElementById(param);
  if (param_element == null) {
    console.log(
      "Unable to find " + parameter_definition_name + " param div container"
    );
    return 1;
  }
  param_input_list = param_element.getElementsByTagName("input");
  if (param_input_list.length == 0) {
    console.log(
      "Unable to find " + parameter_definition_name + " param input/s"
    );
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
      brs = document.getElementById(param).getElementsByTagName("br");
      param_input_list[param_input_list.length - 1].remove();
      param_input_list[param_input_list.length - 1].remove();
      brs[brs.length - 1].remove();
    }
  } else if (param_input_list.length > 1) {
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

//on header input change, execute accordingly
function headerElement(element, param_definitions, config) {
  //obtain section body element
  section_card = element.closest(".section-card");
  if (section_card == null) {
    console.log("Unable to obtain section-card");
    return 1;
  }
  param_container = section_card.getElementsByClassName("section-body");
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
        buildParamContainers("Battery", param_definitions["Battery"], config, [
          "set_use_battery",
        ]);
        element.checked = true;
      } else {
        param_container.innerHTML = "";
      }
      break;

    //if number_of_deferrable_loads, the number of inputs in the "Deferrable Loads" section should add up to number_of_deferrable_loads value in header
    case "number_of_deferrable_loads":
      //get a list of param in section
      param_list = param_container.getElementsByClassName("param");
      if (param_list.length <= 0) {
        console.log(
          "There has been an issue counting the amount of params in number_of_deferrable_loads"
        );
        return 1;
      }
      //calculate how much off the fist parameters input elements amount to is, compering to the number_of_deferrable_loads value
      difference =
        parseInt(element.value) -
        param_container.firstElementChild.querySelectorAll("input").length;
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
  config = {};

  //if section-cards (config sections/list) exists
  config_card = document.getElementsByClassName("section-card");
  //check if page is in list or box view
  config_box_element = document.getElementById("config-box");

  //if true, in list view
  if (Boolean(config_card.length)) {
    //retrieve params and their input/s by looping though param_definitions list
    //loop through the sections
    for (var [section_name, section_object] of Object.entries(
      param_definitions
    )) {
      //loop through parameters
      for (var [
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
          continue;
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
          //obtain param input type from param_definitions, check if param should be formatted as an array
          param_array = Boolean(
            !parameter_definition_object["input"].search("array")
          );

          //build parameters using values extracted from param_inputs

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
        }
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
  const response = await fetch(`set-config`, {
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
  configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //get yaml button
  yaml_button = document.getElementById("yaml");
  if (yaml_button == null) {
    console.log("Unable to obtain yaml button");
  }

  // if section-cards (config sections/list) exists
  config_card = configuration_container.getElementsByClassName("section-card");
  //selected view (0 = box)
  selected_view = Boolean(config_card.length);

  //if default_reset is passed do not switch views, instead reinitialize view with default config as values
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
      loadConfigurationListView(param_definitions, config, list_html);
      yaml_button.style.display = "none";
      break;
    case "list":
      //load box
      loadConfigurationBoxPage(config);
      yaml_button.style.display = "block";
      break;
  }
}

//load box (json textarea) view
async function loadConfigurationBoxPage(config) {
  //get configuration container element
  configuration_container = document.getElementById("configuration-container");
  if (configuration_container == null) {
    errorAlert("Unable to find Configuration Container element");
  }
  //append configuration container with textbox area
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
  if (loading === null) {
    console.log("unable to find loader element");
    return 1;
  }
  if (status === 200 || status === 201) {
    //if status is 200 or 201, then show a tick
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
  config_box_element = document.getElementById("config-box");
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
    response_status = await response.status; //return status
    if (response_status == 201) {
      showChangeStatus(response_status, {});
      blob = await response.blob(); //get data blob
      config = await new Response(blob).json(); //obtain json from blob
      config_box_element.value = JSON.stringify(config, null, 2);
    } else {
      showChangeStatus(response_status, await response.json());
    }
  }
  return 0;
}
