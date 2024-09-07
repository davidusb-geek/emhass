//configuration for dynamically processing index page
//loads either the basic or advance html

//used static files
//advanced.html : template html for displaying all the actions + runtime parameter input
//basic.html : template html for displaying a minimal view of actions

//on page reload get saved data
window.onload = async function () {
  pageSelected = await loadBasicOrAdvanced();

  //add listener for basic and advanced html switch
  document
    .getElementById("basicOrAdvanced")
    .addEventListener("click", () => SwitchBasicOrAdvanced());
};

//add listeners to buttons (based on page)
function loadButtons(page) {
  switch (page) {
    case "advanced":
      [
        "dayahead-optim",
        "forecast-model-fit",
        "forecast-model-predict",
        "forecast-model-tune",
        "regressor-model-fit",
        "regressor-model-predict",
        "perfect-optim",
        "publish-data",
        "naive-mpc-optim",
      ].forEach((id) =>
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
      break;
    case "basic":
      document
        .getElementById("dayahead-optim-basic")
        .addEventListener("click", () => formAction("dayahead-optim", "basic"));
      break;
  }
}

//on check present basic or advanced html inside form element
async function loadBasicOrAdvanced(RequestedPage) {
  let basicFile = "basic.html";
  let advencedFile = "advanced.html";
  var formContainer = document.getElementById("TabSelection"); //container element to house basic or advanced data
  //first check any function  arg
  if (arguments.length == 1) {
    switch (RequestedPage) {
      case "basic":
        htmlData = await getHTMLData(basicFile);
        formContainer.innerHTML = htmlData;
        loadButtons("basic"); //load buttons based on basic or advanced
        if (testStorage()) {
          localStorage.setItem("TabSelection", "basic");
        } //remember mode (save to localStorage)
        return "basic"; //return basic to get saved data
      case "advanced":
        htmlData = await getHTMLData(advencedFile);
        formContainer.innerHTML = htmlData;
        loadButtons("advanced");
        if (testStorage()) {
          localStorage.setItem("TabSelection", "advanced");
        }
        getSavedData();
        return "advanced";
      default:
        htmlData = await getHTMLData(advencedFile);
        formContainer.innerHTML = htmlData;
        loadButtons("advanced");
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
      htmlData = await getHTMLData(advencedFile);
      formContainer.innerHTML = htmlData;
      loadButtons("advanced");
      getSavedData();
      return "advanced";
    } else {
      //else run basic (first time)
      htmlData = await getHTMLData(basicFile);
      formContainer.innerHTML = htmlData;
      loadButtons("basic");
      return "basic";
    }
  } else {
    //if localStorage not supported, set to advanced page
    htmlData = await getHTMLData(advencedFile);
    formContainer.innerHTML = htmlData;
    loadButtons("advanced");
    return "advanced";
  }
}

//on button press, check current displayed page data and switch
function SwitchBasicOrAdvanced() {
  var formContainerChildID =
    document.getElementById("TabSelection").firstElementChild.id;
  if (formContainerChildID === "basic") {
    loadBasicOrAdvanced("advanced");
  } else {
    loadBasicOrAdvanced("basic");
  }
}

//get html data from basic.html or advanced.html
async function getHTMLData(htmlFile) {
  const response = await fetch(`static/` + htmlFile);
  blob = await response.blob(); //get data blob
  htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  return await htmlTemplateData;
}

//function pushing data via post, triggered by button action
async function formAction(action, page) {
  if (page !== "basic") {
    //dont try to get input data in basic mode
    var data = inputToJson(page);
  } else {
    var data = {};
  } //send no data

  if (data !== 0) {
    //don't run if there is an error in the input (box/list) Json data
    showChangeStatus("loading", {}); // show loading div for status
    const response = await fetch(`action/` + action, {
      //fetch data from webserver.py
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data), //note that post can only send data via strings
    });
    if (response.status == 201) {
      showChangeStatus(response.status, {});
      if (page !== "basic") {
        saveStorage(); //save to storage if successful
      }
      return true;
    } //if successful
    else {
      showChangeStatus(response.status, await response.json());
      return false;
    } // else get Log data from response
  } else {
    showChangeStatus("remove"); //replace loading, show tick or cross with none
    return false;
  }
}

//function in control of status icons of post above
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
    getTemplate(); //get updated templates
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
  response = await fetch(`template`, {
    method: "GET",
  });
  blob = await response.blob(); //get data blob
  htmlTemplateData = await new Response(blob).text(); //obtain html from blob
  templateDiv = document.getElementById("template"); //get template container element to override
  templateDiv.innerHTML = htmlTemplateData; //override container inner html with new data
  var scripts = Array.from(templateDiv.getElementsByTagName("script")); //replace script tags manually
  for (const script of scripts) {
    var TempScript = document.createElement("script");
    TempScript.innerHTML = script.innerHTML;
    script.parentElement.appendChild(TempScript);
  }
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
  return false;
}

//function gets saved data (if any)
function getSavedData() {
  dictInputs(); //check selected current (List or Box) is correct
  if (testStorage()) {
    //if local storage exists and works
    let selectElement = document.getElementById("input-select"); // select button element
    var input_container = document.getElementById("input-container"); // container div containing all dynamic input elements (Box/List)
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
        storedJson = JSON.parse(
          localStorage.getItem("input_container_content")
        );
        if (Object.keys(storedJson).length > 0) {
          input_container.innerHTML = "";
          i = 1;
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
  var data = JSON.stringify(inputToJson());
  if (testStorage() && data != "{}") {
    //don't bother saving if empty and/or storage don't exist
    localStorage.setItem("input_container_content", data);
  }
}

//function gets values from input-list/text-area(from input-box) elements and return json dict object
function inputToJson() {
  var input_container = document.getElementById("input-container"); //container
  let inputListArr = document.getElementsByClassName("input-list"); //list
  let inputTextArea = document.getElementById("text-area"); //box
  let input_container_child = null;
  input_container_child = input_container.firstElementChild; //work out which element is first inside container div
  var jsonReturnData = {};

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
      var value =
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
  var input_container = document.getElementById("input-container"); // container div containing all dynamic input elements
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
  var input_container = document.getElementById("input-container");
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
