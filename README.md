<div align="center">
  <br>
  <img alt="EMHASS" src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/logo_docs.png" width="700px">
  <h1>Energy Management for Home Assistant</h1>
  <strong></strong>
</div>
<br>

<p align="center">
  <a style="text-decoration:none" href="https://pypi.org/project/emhass/">
    <img alt="PyPI - Version" src="https://img.shields.io/pypi/v/emhass">
  </a>
  <a style="text-decoration:none" href="https://anaconda.org/channels/davidusb/packages/emhass/overview">
    <img alt="Conda - Version" src="https://img.shields.io/conda/v/davidusb/emhass">
  </a>
  <a style="text-decoration:none" href="https://github.com/davidusb-geek/emhass/actions">
    <img alt="EMHASS GitHub Workflow Status" src="https://github.com/davidusb-geek/emhass/actions/workflows/publish_docker.yaml/badge.svg?event=release">
  </a>
  <a style="text-decoration:none" href="https://github.com/davidusb-geek/emhass/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/davidusb-geek/emhass">
  </a>
  <a style="text-decoration:none" href="https://pypi.org/project/emhass/">
    <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/emhass">
  </a>
  <a style="text-decoration:none" href="https://pypi.org/project/emhass/">
    <img alt="PyPI - Status" src="https://img.shields.io/pypi/status/emhass">
  </a>
  <a style="text-decoration:none" href="https://emhass.readthedocs.io/en/latest/">
    <img alt="Read the Docs" src="https://img.shields.io/readthedocs/emhass">
  </a>
    <a hstyle="text-decoration:none" ref="https://codecov.io/github/davidusb-geek/emhass" >
    <img src="https://codecov.io/github/davidusb-geek/emhass/branch/master/graph/badge.svg?token=BW7KSCHN90"/>
  </a>
  <a hstyle="text-decoration:none" ref="https://github.com/davidusb-geek/emhass/actions/workflows/codeql.yml" >
    <img src="https://github.com/davidusb-geek/emhass/actions/workflows/codeql.yml/badge.svg?branch=master&event=schedule"/>
  </a>
  <a style="text-decoration:none" href="https://sonarcloud.io/summary/new_code?id=davidusb-geek_emhass">
    <img alt="SonarQube security rating" src="https://sonarcloud.io/api/project_badges/measure?project=davidusb-geek_emhass&metric=security_rating">
  </a>
  <a style="text-decoration:none" href="https://sonarcloud.io/summary/new_code?id=davidusb-geek_emhass">
    <img alt="SonarQube security Vulnerabilities" src="https://sonarcloud.io/api/project_badges/measure?project=davidusb-geek_emhass&metric=vulnerabilities">
  </a>
  <a style="text-decoration:none" href="https://sonarcloud.io/summary/new_code?id=davidusb-geek_emhass">
    <img alt="SonarQube reliability" src="https://sonarcloud.io/api/project_badges/measure?project=davidusb-geek_emhass&metric=reliability_rating">
  </a>
  <a style="text-decoration:none" href="https://sonarcloud.io/summary/new_code?id=davidusb-geek_emhass">
    <img alt="SonarQube bugs" src="https://sonarcloud.io/api/project_badges/measure?project=davidusb-geek_emhass&metric=bugs">
  </a>
  
</p>

<div align="center">
 <a style="text-decoration:none" href="https://emhass.readthedocs.io/en/latest/">
      <img src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/Documentation_button.svg" alt="Documentation">
  </a>
   <a style="text-decoration:none" href="https://community.home-assistant.io/t/emhass-an-energy-management-for-home-assistant/338126">
      <img src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/Community_button.svg" alt="Community">
  </a>
  <a style="text-decoration:none" href="https://github.com/davidusb-geek/emhass/issues">
      <img src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/Issues_button.svg" alt="Issues">
  </a>
  <a style="text-decoration:none" href="https://github.com/davidusb-geek/emhass-add-on">
     <img src="https://raw.githubusercontent.com/davidusb-geek/emhass/master/docs/images/EMHASS_Add_on_button.svg" alt="EMHASS Add-on">
  </a>
</div>

<br>
<p align="left">
EMHASS is a Python module designed to optimize your home energy interfacing with Home Assistant.
</p>

## Introduction

EMHASS (Energy Management for Home Assistant) is an optimization tool designed for residential households. The package uses a Linear Programming approach to optimize energy usage while considering factors such as electricity prices, power generation from solar panels, and energy storage from batteries. EMHASS provides a high degree of configurability, making it easy to integrate with Home Assistant and other smart home systems. Whether you have solar panels, energy storage, or just a controllable load, EMHASS can provide an optimized daily schedule for your devices, allowing you to save money and minimize your environmental impact.

The complete documentation for this package is [available here](https://emhass.readthedocs.io/en/latest/).

To get started you can follow our [🚀 Quick Start](/docs/quick_start.md) guide in the documentation.

Here are the guides for:
- [📦 Installation methods](/docs/installation_methods.md)
- [📖 Usage](/docs/usage_guide.md)
- [🤖 Home Assistant Automations](/docs/automations.md)

## Development

Pull requests are very much accepted on this project. For development, you can find some instructions here [Development](/docs/develop.md).

## License

MIT License

Copyright (c) 2021-2025 David HERNANDEZ

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
