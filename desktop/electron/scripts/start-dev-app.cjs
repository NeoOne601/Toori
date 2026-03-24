const fs = require("fs");
const path = require("path");
const { spawnSync, spawn } = require("child_process");

const rootDir = path.resolve(__dirname, "..");
const electronApp = path.join(rootDir, "node_modules", "electron", "dist", "Electron.app");
const devBundleRoot = path.join(rootDir, ".dev-app");
const devApp = path.join(devBundleRoot, "Toori Lens Assistant.app");
const plistPath = path.join(devApp, "Contents", "Info.plist");
const bundleId = "ai.toori.lensassistant.dev";

function ensureDirectory(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function copyBundle() {
  ensureDirectory(devBundleRoot);
  fs.rmSync(devApp, { recursive: true, force: true });
  fs.cpSync(electronApp, devApp, { recursive: true });
}

function patchInfoPlist() {
  let plist = fs.readFileSync(plistPath, "utf8");
  const replacements = new Map([
    ["<string>Electron</string>", "<string>Toori Lens Assistant</string>"],
    ["<string>com.github.Electron</string>", `<string>${bundleId}</string>`],
    ["<string>This app needs access to the camera</string>", "<string>Toori Lens Assistant needs camera access for live lens preview and capture.</string>"],
    ["<string>This app needs access to the microphone</string>", "<string>Toori Lens Assistant optionally uses microphone access for future multimodal capture.</string>"],
  ]);
  for (const [from, to] of replacements) {
    plist = plist.replace(from, to);
  }
  if (!plist.includes("<key>CFBundleDisplayName</key>")) {
    plist = plist.replace(
      "<key>CFBundleName</key>\n    <string>Toori Lens Assistant</string>",
      "<key>CFBundleDisplayName</key>\n    <string>Toori Lens Assistant</string>\n    <key>CFBundleName</key>\n    <string>Toori Lens Assistant</string>",
    );
  }
  fs.writeFileSync(plistPath, plist);
}

function isSignableFile(filePath) {
  const stat = fs.statSync(filePath);
  if (!stat.isFile()) {
    return false;
  }
  if (/\.(dylib|so|node)$/.test(filePath)) {
    return true;
  }
  return (stat.mode & 0o111) !== 0;
}

function signPath(target) {
  const result = spawnSync("/usr/bin/codesign", ["--force", "--sign", "-", "--timestamp=none", target], {
    stdio: "inherit",
  });
  if (result.status !== 0) {
    throw new Error(`codesign failed for ${target}`);
  }
}

function signExecutableTree(baseDir) {
  if (!fs.existsSync(baseDir)) {
    return;
  }
  for (const entry of fs.readdirSync(baseDir, { withFileTypes: true })) {
    const entryPath = path.join(baseDir, entry.name);
    if (entry.isDirectory()) {
      signExecutableTree(entryPath);
      continue;
    }
    if (isSignableFile(entryPath)) {
      signPath(entryPath);
    }
  }
}

function signBundle() {
  const contentsDir = path.join(devApp, "Contents");
  const frameworksDir = path.join(contentsDir, "Frameworks");
  signPath(path.join(contentsDir, "MacOS", "Electron"));
  signExecutableTree(frameworksDir);
  signPath(path.join(frameworksDir, "Electron Helper.app"));
  signPath(path.join(frameworksDir, "Electron Helper (GPU).app"));
  signPath(path.join(frameworksDir, "Electron Helper (Plugin).app"));
  signPath(path.join(frameworksDir, "Electron Helper (Renderer).app"));
  signPath(devApp);
}

function launchBundle() {
  const child = spawn("/usr/bin/open", ["-na", devApp, "--args", rootDir], {
    stdio: "inherit",
    env: process.env,
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });
}

function launchFallback() {
  const cli = path.join(rootDir, "node_modules", "electron", "cli.js");
  const child = spawn(process.execPath, [cli, rootDir], {
    stdio: "inherit",
    env: process.env,
  });
  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }
    process.exit(code ?? 0);
  });
}

if (process.platform !== "darwin") {
  launchFallback();
} else {
  copyBundle();
  patchInfoPlist();
  signBundle();
  launchBundle();
}
