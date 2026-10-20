// js_stubs.js — minimal browser shims for evaluating the inline site
// script under node (web/test_site_js.py prepends this file to the
// <script> block extracted from web/site/index.html).
//
// Only what the script touches at evaluation time: element lookups,
// tab/expander wiring, and the four fetch() calls (rejected, so the
// page ends in its error path and pure helpers stay callable).
//
// Limitation: these shims only support evaluating the script's pure
// helpers (tooltip sentence construction, formatters, guards) plus the
// built-element shape of endpointRow()/baselineNote() — appendChild
// records children so a driver can walk the disclosure DOM the script
// built. Tooltip delivery (the td.title assignment inside render),
// event wiring, and the live fetch flows are not exercised: fetch()
// rejects so render() never runs, getElementById hands out a fresh
// stub per call, and there is no querySelector.

function stubElement() {
  return {
    textContent: "",
    hidden: false,
    disabled: false,
    title: "",
    className: "",
    colSpan: 0,
    tabIndex: 0,
    children: [],
    dataset: {},
    style: {},
    classList: {
      toggle: function () {},
      add: function () {},
      remove: function () {},
    },
    addEventListener: function () {},
    appendChild: function (child) {
      this.children.push(child);
      return child;
    },
    remove: function () {},
    select: function () {},
  };
}

var window = { location: { href: "http://127.0.0.1:8734/index.html" } };
var document = {
  getElementById: function () {
    return stubElement();
  },
  createElement: function () {
    return stubElement();
  },
  querySelectorAll: function () {
    return [];
  },
  querySelector: function () {
    return null;
  },
  body: stubElement(),
};
var navigator = {};
var fetch = function () {
  return Promise.reject(new Error("stub: fetch disabled in JS tests"));
};
