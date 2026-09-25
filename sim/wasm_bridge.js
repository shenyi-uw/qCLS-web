"use strict";

const path = require("node:path");
const readline = require("node:readline");

const moduleArgument = process.argv.indexOf("--module");
if (moduleArgument < 0 || !process.argv[moduleArgument + 1]) {
    throw new Error("Expected --module <qcls_core.js>");
}

let resolveRuntime;
const runtimeReady = new Promise((resolve) => {
    resolveRuntime = resolve;
});
const Module = require(path.resolve(process.argv[moduleArgument + 1]));
Module.onRuntimeInitialized = resolveRuntime;

function readStimulus(selector, minLevel, maxLevel) {
    const freqPointer = Module._malloc(4);
    const levelPointer = Module._malloc(4);
    try {
        selector(minLevel, maxLevel, freqPointer, levelPointer);
        const frequency = new Float32Array(Module.HEAPF32.buffer, freqPointer, 1)[0];
        const level = new Float32Array(Module.HEAPF32.buffer, levelPointer, 1)[0];
        return { frequency, level };
    } finally {
        Module._free(freqPointer);
        Module._free(levelPointer);
    }
}

function runFit(message) {
    const { frequencies, levels, responses } = message;
    if (!Array.isArray(frequencies) || frequencies.length === 0 ||
        levels.length !== frequencies.length || responses.length !== frequencies.length) {
        throw new Error("fit requires equally sized non-empty frequencies, levels, and responses");
    }

    const count = frequencies.length;
    const frequencyPointer = Module._malloc(count * 4);
    const levelPointer = Module._malloc(count * 4);
    const responsePointer = Module._malloc(count * 4);
    const outputPointer = Module._malloc(100 * 4);
    try {
        new Float32Array(Module.HEAPF32.buffer, frequencyPointer, count).set(frequencies);
        new Float32Array(Module.HEAPF32.buffer, levelPointer, count).set(levels);
        new Float32Array(Module.HEAPF32.buffer, responsePointer, count).set(responses);
        const ok = Module._qcls_pca_fit_report(
            frequencyPointer, levelPointer, responsePointer, count, outputPointer
        );
        if (!ok) throw new Error("qcls_pca_fit_report failed");
        return Array.from(new Float32Array(Module.HEAPF32.buffer, outputPointer, 100));
    } finally {
        Module._free(frequencyPointer);
        Module._free(levelPointer);
        Module._free(responsePointer);
        Module._free(outputPointer);
    }
}

async function dispatch(message) {
    await runtimeReady;
    switch (message.cmd) {
        case "init":
            Module._init_bayesian_state();
            return { ok: true };
        case "update":
            Module._qcls_update_trial(message.frequency, message.level, message.response);
            return { ok: true };
        case "bayesian":
            return readStimulus(
                Module._qcls_select_bayesian_next,
                message.min_level,
                message.max_level
            );
        case "isophon":
            return readStimulus(
                Module._qcls_select_isophon_next,
                message.min_level,
                message.max_level
            );
        case "fit":
            return { boundaries: runFit(message) };
        default:
            throw new Error(`Unknown bridge command: ${message.cmd}`);
    }
}

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let pending = Promise.resolve();
input.on("line", (line) => {
    pending = pending.then(async () => {
        try {
            const result = await dispatch(JSON.parse(line));
            process.stdout.write(JSON.stringify(result) + "\n");
        } catch (error) {
            process.stdout.write(JSON.stringify({ ok: false, error: String(error) }) + "\n");
        }
    });
});